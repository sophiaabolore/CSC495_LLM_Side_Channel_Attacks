import subprocess
import os
import time

# Define the commands to run for each repository
speculative_sampling_command = [
    'python', 'main.py',
    '--input', 'The quick brown fox jumps over the lazy dog',
    '--target_model_name', 'bigscience/bloomz-7b1',
    '--approx_model_name', 'bigscience/bloom-560m'
]

lookahead_decoding_command = [
    'USE_LADE=1', 'python', 'applications/chatbot.py',
    '--model_path', 'meta-llama/Llama-2-7b-chat-hf', '--debug', '--chat'
]

# Define the paths to each repository
speculative_sampling_path = '/path/to/LLMSpeculativeSampling'
lookahead_decoding_path = '/path/to/LookaheadDecoding'

def measure_execution_time(command, working_directory):
    # Change the working directory
    os.chdir(working_directory)
    
    # Start the timer
    start_time = time.time()
    
    # Run the command
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    
    # Calculate the elapsed time
    elapsed_time = time.time() - start_time
    
    # Return the result and the elapsed time
    return result, elapsed_time

# Measure Speculative Sampling
spec_sampling_result, spec_sampling_time = measure_execution_time(
    speculative_sampling_command, speculative_sampling_path
)
print(f"Speculative Sampling Execution Time: {spec_sampling_time} seconds")
print("Output:", spec_sampling_result.stdout.decode())

# Measure Lookahead Decoding
lookahead_result, lookahead_time = measure_execution_time(
    lookahead_decoding_command, lookahead_decoding_path
)
print(f"Lookahead Decoding Execution Time: {lookahead_time} seconds")
print("Output:", lookahead_result.stdout.decode())
