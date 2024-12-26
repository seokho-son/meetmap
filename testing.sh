#!/bin/bash

# Default URL to be requested
default_url="http://127.0.0.1:1111/room/7-563"

# Prompt the user for the URL
read -p "Enter the URL to be requested (default: $default_url): " url
url=${url:-$default_url}

# Prompt the user for the maximum batch size
read -p "Enter the maximum batch size (default: 1000): " max_batch_size
max_batch_size=${max_batch_size:-1000}

# Temporary file to store results
temp_file=$(mktemp)

# Function to perform a single request and print the result
perform_request() {
    local request_number=$1
    local url=$2
    local result=$(curl -s -o /dev/null -w "%{http_code} %{time_total}" "$url")
    local http_code=$(echo $result | awk '{print $1}')
    local time_total=$(echo $result | awk '{print $2}')
    echo "$request_number $http_code $time_total" >> "$temp_file"
}

export -f perform_request
export url
export temp_file

# Array to store summary results
declare -a summaries

# Perform tests with different batch sizes
for batch_size in $(seq 100 100 $max_batch_size); do
    echo ""
    echo "Performing test with batch size: $batch_size"
    temp_file=$(mktemp)
    
    # Perform requests in parallel
    seq 1 $batch_size | xargs -P$batch_size -I{} bash -c 'perform_request "$@"' _ {} "$url"

    # Calculate and print summary for the current batch size
    total_time=$(awk '{sum += $3} END {print sum}' "$temp_file")
    average_time=$(awk '{sum += $3} END {print sum/NR}' "$temp_file")
    min_time=$(awk 'NR == 1 || $3 < min {min = $3} END {print min}' "$temp_file")
    max_time=$(awk 'NR == 1 || $3 > max {max = $3} END {print max}' "$temp_file")
    success_count=$(awk '$2 == 200 {count++} END {print count}' "$temp_file")
    failure_count=$(awk '$2 != 200 {count++} END {print count}' "$temp_file")

    echo ""
    echo "Summary for batch size $batch_size:"
    echo "---------------------------------------------"
    echo "Total Requests: $batch_size"
    echo "Successful Requests: $success_count"
    echo "Failed Requests: $failure_count"
    echo "Total Time: $total_time seconds"
    echo "Average Time: $average_time seconds"
    echo "Min Time: $min_time seconds"
    echo "Max Time: $max_time seconds"

    # Store summary results
    summaries+=("$batch_size $average_time $min_time $max_time")

    # Clean up
    rm "$temp_file"
done

# Print final summary
echo ""
echo "Final Summary:"
printf "%-12s %-18s %-12s %-12s\n" "Batch Size" "Average Time (s)" "Min Time (s)" "Max Time (s)"
echo "----------------------------------------------------------"
for summary in "${summaries[@]}"; do
    printf "%-12s %-18s %-12s %-12s\n" $summary
done