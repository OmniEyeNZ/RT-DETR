#%%
import os
import tensorrt as trt
import numpy as np
#%%
ONNX_SIM_MODEL_PATH = 'models/2024-09-13.r18vd.64.onnx'
TENSORRT_ENGINE_PATH_PY = 'models/2024-09-13.r18vd.64.trt.engine'

#%%
# View shape of network
def view_network_shape(network):
    print("\n--- Network Inputs ---")
    for i in range(network.num_inputs):
        input_tensor = network.get_input(i)
        print(f"Input {i}: Name={input_tensor.name}, Shape={input_tensor.shape}")
    print("\n--- Network Outputs ---")
    for i in range(network.num_outputs):
        output_tensor = network.get_output(i)
        print(f"Output {i}: Name={output_tensor.name}, Shape={output_tensor.shape}")
#%%
# Save tensorrt model
def save_engine(engine,  engine_file_path):
    if engine is not None:
        with open(engine_file_path, "wb") as f:
            f.write(engine)
        print(f"Engine successfully saved to {engine_file_path}")
    else:
        print("Failed to build the engine.")
#%%
# Convert .onnx omdel to tensorrt model
def onnx_to_tensorrt(onnx_file_path = ONNX_SIM_MODEL_PATH, engine_file_path = TENSORRT_ENGINE_PATH_PY, flop=16):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    
    # view shape
    view_network_shape(network)

    # Load .onnx model
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_file_path, 'rb') as model_file:
        if not parser.parse(model_file.read()):
            print('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
    print("Completed parsing ONNX file")

    # Create config element
    config = builder.create_builder_config()
    # Set workspace memory size
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 1GB workspace

    # Set precision
    if flop == 16:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("FP16 mode enabled")
    elif flop == 32:
        print("FP32 mode enabled")

    # Create optimization profile
    profile = builder.create_optimization_profile()

    # Set the optimization profile for the input "images".
    input_tensor_0 = network.get_input(0)
    input_shape_0 = input_tensor_0.shape
    input_name_0 = input_tensor_0.name

    # min_shape_0 = (1, *input_shape_0[1:])
    # opt_shape_0 = (4, *input_shape_0[1:])
    # max_shape_0 = (8, *input_shape_0[1:])
    print("------input and shape-------",input_name_0,input_shape_0)

    profile.set_shape(input_name_0, input_shape_0, input_shape_0, input_shape_0)

    config.add_optimization_profile(profile)
    print("Building TensorRT engine. This may take a few minutes...")

    # start build engine
    serialized_engine = builder.build_serialized_network(network, config)

    # save engine
    save_engine(serialized_engine, engine_file_path)
#%%
onnx_to_tensorrt(flop=16)
# %%
