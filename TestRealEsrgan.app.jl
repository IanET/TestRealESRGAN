using LibOnnxRuntime, Images, FileIO
import .GC: @preserve
import .Base: cconvert

const QnnRuntime = ".\\microsoft.ml.onnxruntime.qnn.1.23.2\\runtimes\\win-arm64\\native\\onnxruntime.dll"

to_cwstring(s::String) = cconvert(Cwstring, s)

if Sys.iswindows()
    const MODEL_PATH = "Real-ESRGAN-x4plus.onnx" |> to_cwstring
elseif Sys.islinux()
    const MODEL_PATH = "Real-ESRGAN-x4plus.onnx"
end

function check_status(ort, status)
    if status != OrtStatusPtr(0)
        msg = GetErrorMessage(ort, status) |> unsafe_string
        code = GetErrorCode(ort, status)
        # println("Status: $code $msg")
        @assert false "ONNX Runtime returned an error: $code $msg"
        ReleaseStatus(ort, status)
    end
end

# base = OrtGetApiBase() |> unsafe_load

# Load custom ONNX Runtime instead of default one
# TODO Assert running on Snapdragon with QNN support
base = @ccall(QnnRuntime.OrtGetApiBase()::Ptr{OrtApiBase}) |> unsafe_load

ort = GetApi(base, ORT_API_VERSION) |> unsafe_load
env = Ptr{OrtEnv}() |> Ref
# status = CreateEnv(ort, ORT_LOGGING_LEVEL_VERBOSE, "Test", env)
status = CreateEnv(ort, ORT_LOGGING_LEVEL_ERROR, "Test", env)
check_status(ort, status)
@info "CreateEnv" status env[]

options = Ptr{OrtSessionOptions}() |> Ref
status = CreateSessionOptions(ort, options)
check_status(ort, status)
@info "CreateSessionOptions" status options[]

provider_name = "QNNExecutionProvider"
provider_option_keys = ["backend_type"]
provider_option_values = ["gpu"]
@preserve provider_option_keys provider_option_values begin
    status = SessionOptionsAppendExecutionProvider(ort, options[], provider_name, pointer.(provider_option_keys), pointer.(provider_option_values), length(provider_option_keys))
    check_status(ort, status)
end
@info "Enabled QNN GPU execution provider"

session = Ptr{OrtSession}() |> Ref
status = CreateSession(ort, env[], MODEL_PATH, options[], session)
check_status(ort, status)
@info "CreateSession" status session[]

allocator = Ptr{OrtAllocator}() |> Ref
status = GetAllocatorWithDefaultOptions(ort, allocator)
check_status(ort, status)
@info "GetAllocatorWithDefaultOptions" status allocator[]

memory_info = Ptr{OrtMemoryInfo}() |> Ref
status = CreateCpuMemoryInfo(ort, OrtArenaAllocator, OrtMemTypeDefault, memory_info)
check_status(ort, status)
@info "CreateCpuMemoryInfo" status memory_info[]

img = load("Sample2.jpg") 
@info "Input image shape:" size(img) # 128x128
cv = channelview(img)
cvf32 = Float32.(cv) # N0f8 is 0.0 - 1.0
cvf32p = permutedims(cvf32, (2, 3, 1)) |> collect # 3x128x128 in row-major order

@info "Input array shape:" size(cvf32p) 

input_shape = Clonglong[1, 3, 128, 128]
# input_values = fill(Cfloat(1.0), 5, 4, 3) # ONNX uses row-major order
input_tensor = Ptr{OrtValue}() |> Ref   
status = CreateTensorWithDataAsOrtValue(ort, memory_info[], cvf32p, sizeof(cvf32p), input_shape, length(input_shape), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, input_tensor)
check_status(ort, status)
@info "CreateTensorWithDataAsOrtValue" status input_tensor[]

input_tensors = [input_tensor[]]
output_tensor = Ptr{OrtValue}() 
output_tensors = [output_tensor]
num_outputs = length(output_tensors)
output_name = "upscaled_image"
input_names = ["image"]
output_names = ["upscaled_image"]
output_tensors = Ptr{OrtValue}() |> Ref

@preserve input_names output_names begin 
    @time status = Run(ort, session[], C_NULL, pointer.(input_names), input_tensors, length(input_tensors), pointer.(output_names), length(output_names), output_tensors)
end
check_status(ort, status)   
@info "Run" status output_tensors[]

out = Ptr{Cvoid}() |> Ref
status = GetTensorMutableData(ort, output_tensors[], out)
check_status(ort, status)
@info "GetTensorMutableData" status out[]

output_shape = Clonglong[1, 3, 512, 512]
output_array = unsafe_wrap(Array, out[] |> Ptr{Cfloat}, prod(output_shape))

# The output from ONNX is a flat, row-major array. Reshape it and convert it for Julia.
# Just ignore the batch dimension (1).
output_row_major = reshape(output_array, (512, 512, 3))
output_col_major = permutedims(output_row_major, (3, 1, 2))
output_n0f8 = N0f8.(clamp.(output_col_major, 0.0f0, 1.0f0))
img_out = colorview(RGB, output_n0f8)
# save("Sample2_upscaled.png", img_out)


# Clean up resources
ReleaseValue(ort, input_tensor[])
ReleaseValue(ort, output_tensors[])
ReleaseSession(ort, session[])
ReleaseSessionOptions(ort, options[])
ReleaseEnv(ort, env[])

@info "Test completed successfully."