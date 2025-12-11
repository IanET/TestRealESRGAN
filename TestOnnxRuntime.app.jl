using LibOnnxRuntime
import .GC: @preserve
import .Base: cconvert

to_cwstring(s::String) = cconvert(Cwstring, s)

if Sys.iswindows()
    const MODEL_PATH = "model.onnx" |> to_cwstring
elseif Sys.islinux()
    const MODEL_PATH = "model.onnx"
end
const INPUT_NAME = "x"
const OUTPUT_NAME = "y"

function check_status(ort, status)
    if status != OrtStatusPtr(0)
        msg = GetErrorMessage(ort, status) |> unsafe_string
        code = GetErrorCode(ort, status)
        # println("Status: $code $msg")
        @assert false "ONNX Runtime returned an error: $code $msg"
        ReleaseStatus(ort, status)
    end
end

base = OrtGetApiBase() |> unsafe_load
ort = GetApi(base, ORT_API_VERSION) |> unsafe_load
env = Ptr{OrtEnv}() |> Ref
status = CreateEnv(ort, ORT_LOGGING_LEVEL_VERBOSE, "Test", env)
check_status(ort, status)
@info "CreateEnv" status env[]

options = Ptr{OrtSessionOptions}() |> Ref
status = CreateSessionOptions(ort, options)
check_status(ort, status)
@info "CreateSessionOptions" status options[]

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

input_shape = Clonglong[3, 4, 5]
input_values = fill(Cfloat(1.0), 5, 4, 3) # ONNX uses row-major order
input_tensor = Ptr{OrtValue}() |> Ref   
status = CreateTensorWithDataAsOrtValue(ort, memory_info[], input_values, sizeof(input_values), input_shape, length(input_shape), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, input_tensor)
check_status(ort, status)
@info "CreateTensorWithDataAsOrtValue" status input_tensor[]

input_tensors = [input_tensor[]]
output_tensor = Ptr{OrtValue}() 
output_tensors = [output_tensor]
num_outputs = length(output_tensors)
input_names = [pointer(INPUT_NAME)] # Needs GC preserve
output_names = [pointer(OUTPUT_NAME)] # Needs GC preserve
output_tensors = Ptr{OrtValue}() |> Ref

@preserve INPUT_NAME OUTPUT_NAME begin 
    status = Run(ort, session[], C_NULL, input_names, input_tensors, length(input_tensors), output_names, length(output_names), output_tensors)
end
check_status(ort, status)   
@info "Run" status output_tensors[]

out = Ptr{Cvoid}() |> Ref
status = GetTensorMutableData(ort, output_tensors[], out)
check_status(ort, status)
@info "GetTensorMutableData" status out[]

output_array = unsafe_wrap(Array, out[] |> Ptr{Cfloat}, prod(input_shape)) |> v -> reshape(v, (5, 4, 3))
@info "Output values:" output_array
@assert isapprox.(output_array, 0.731) |> all

# Clean up resources
ReleaseValue(ort, input_tensor[])
ReleaseValue(ort, output_tensors[])
ReleaseSession(ort, session[])
ReleaseSessionOptions(ort, options[])
ReleaseEnv(ort, env[])

@info "Test completed successfully."