import streamlit as st
import numpy as np
import time
import random
import pandas as pd
import io

class AlgorithmExplanations:
    @staticmethod
    def get_explanation(algo_name):
        explanations = {
    "Find Max/Min": """
    ### Maximum and Minimum Finding Algorithm

    This algorithm finds both the maximum and minimum values in an array with fewer comparisons than the brute-force approach by using a divide-and-conquer strategy.
    
    #### How it Works:
    1. **Divide** the array into two halves.
    2. **Recursively** find the max and min values in each half.
    3. **Combine** results by comparing the max and min values from each half to get the overall max and min.

    #### Time Complexity:
    - Recursive relation: T(n) = 2T(n/2) + 2 comparisons.
    - Solves to approximately **3n/2 - 2 comparisons** for an array of n elements.

    #### Why Use It?
    - **Efficiency**: Compared to brute force, which uses 2(n-1) comparisons, this approach only requires about 3n/2 - 2 comparisons.
    """,
    
    "Exponentiation": """
    ### Fast Exponentiation Algorithm (Exponentiation by Squaring)
    
    This algorithm efficiently computes powers (a^n) using the divide-and-conquer technique. Rather than multiplying `a` by itself `n` times, it reduces the problem size by taking advantage of even and odd exponents.

    #### How it Works:
    1. **If `n` is even**: Compute a^(n/2) once and then square the result.
    2. **If `n` is odd**: Compute a^(n-1) and then multiply by `a`.

    #### Time Complexity:
    - Reduces to **O(log n)**, as each recursive call reduces `n` by half.

    #### Why Use It?
    - **Performance**: Brute-force multiplication takes n-1 multiplications, while this approach takes only log(n) multiplications.
    """,
    
    "Count Inversions": """
    ### Inversion Counting Algorithm

    This algorithm counts the number of "inversions" in an array—pairs of elements where a larger element appears before a smaller one. It uses a modified merge sort approach to count inversions more efficiently than a brute-force double loop.

    #### How it Works:
    1. **Divide** the array into two halves.
    2. **Count inversions** in the left half and the right half recursively.
    3. **Count split inversions** between the two halves during the merge step.

    #### Time Complexity:
    - **O(n log n)** due to the divide-and-conquer approach, similar to merge sort.

    #### Applications:
    - **Sorting distance**: Measures how far an array is from being sorted.
    - **Rank correlation**: Used in comparing rankings or orderings in statistics.
    """,
    
    "Quicksort": """
    ### Quicksort Algorithm

    Quicksort is a highly efficient, in-place sorting algorithm that uses partitioning and recursion to sort an array. It is a classic example of divide and conquer, where the array is divided based on a pivot element.

    #### How it Works:
    1. **Choose a pivot** element.
    2. **Partition** the array so that all elements less than the pivot are on the left, and all elements greater are on the right.
    3. **Recursively apply** quicksort to the subarrays on both sides of the pivot.

    #### Time Complexity:
    - **Average Case**: O(n log n)
    - **Worst Case**: O(n²), though this is rare if the pivot is chosen well.

    #### When to Use:
    - Ideal for **random or large datasets**.
    - **In-place sorting**: Requires less memory compared to merge sort.
    - **Good cache performance** due to sequential memory access.
    """,
    
    "Closest Pair 1D": """
    ### Closest Pair in 1D Algorithm

    This algorithm finds the smallest difference between any two numbers in a 1D array. It leverages sorting to quickly compare only adjacent elements, which reduces the number of comparisons.

    #### How it Works:
    1. **Sort** the array: O(n log n).
    2. **Scan** the sorted array, comparing each adjacent pair to find the smallest difference: O(n).

    #### Time Complexity:
    - **O(n log n)**, mainly due to the sorting step.

    #### Applications:
    - **Data similarity**: Identifying closely related numbers in a dataset.
    - **Pattern recognition**: Useful in detecting near-duplicates.
    - **Time series analysis**: Finds closely spaced events.
    """,
    
    "Karatsuba Multiplication": """
    ### Karatsuba Multiplication Algorithm

    The Karatsuba algorithm efficiently multiplies large numbers by breaking down the multiplication process using a divide-and-conquer approach. It reduces the number of recursive multiplications needed, making it faster than the traditional grade-school multiplication.

    #### How it Works:
    1. **Split** each number into two halves (high and low parts).
    2. **Recursively compute** three products instead of four (Karatsuba's trick).
    3. **Combine** the results with appropriate shifts to get the final product.

    #### Time Complexity:
    - **O(n^log₂3) ≈ O(n^1.585)**, significantly faster than the **O(n²)** complexity of traditional multiplication.

    #### Why Use It?
    - More efficient for **large numbers** (hundreds of digits).
    - Widely used in **cryptography** and **computer arithmetic**.
    """
}

        return explanations.get(algo_name, "Explanation not available")

class DivideAndConquerAlgorithms:
    @staticmethod
    def find_max_min(arr, left, right):
        if left == right:
            return arr[left], arr[left]
        elif right == left + 1:
            return (max(arr[left], arr[right]), min(arr[left], arr[right]))

        mid = (left + right) // 2
        max1, min1 = DivideAndConquerAlgorithms.find_max_min(arr, left, mid)
        max2, min2 = DivideAndConquerAlgorithms.find_max_min(arr, mid + 1, right)
        
        return max(max1, max2), min(min1, min2)

    @staticmethod
    def power(base, exp):
        if exp == 0:
            return 1
        elif exp % 2 == 0:
            half = DivideAndConquerAlgorithms.power(base, exp // 2)
            return half * half
        else:
            return base * DivideAndConquerAlgorithms.power(base,exp - 1)

    @staticmethod
    def count_inversions(arr):
        def merge_and_count(arr, temp, left, mid, right):
            i = left    # Starting index for left subarray
            j = mid + 1 # Starting index for right subarray
            k = left    # Starting index to be sorted
            inv_count = 0
            
            while i <= mid and j <= right:
                if arr[i] <= arr[j]:
                    temp[k] = arr[i]
                    i += 1
                else:
                    temp[k] = arr[j]
                    inv_count += (mid - i + 1)  # All remaining elements in the left subarray are greater
                    j += 1
                k += 1
            
            # Copy remaining elements of the left subarray, if any
            while i <= mid:
                temp[k] = arr[i]
                i += 1
                k += 1
            
            # Copy remaining elements of the right subarray, if any
            while j <= right:
                temp[k] = arr[j]
                j += 1
                k += 1
            
            # Copy sorted elements back into the original array
            for i in range(left, right + 1):
                arr[i] = temp[i]
            
            return inv_count

        def merge_sort_and_count(arr, temp, left, right):
            inv_count = 0
            if left < right:
                mid = (left + right) // 2
                
                inv_count += merge_sort_and_count(arr, temp, left, mid)
                inv_count += merge_sort_and_count(arr, temp, mid + 1, right)
                inv_count += merge_and_count(arr, temp, left, mid, right)
            
            return inv_count
        
        n = len(arr)
        temp = [0] * n
        return merge_sort_and_count(arr, temp, 0, n - 1)


    @staticmethod
    def quicksort(arr, low, high):
        if low < high:
            pivot_index = DivideAndConquerAlgorithms.partition(arr, low, high)
            DivideAndConquerAlgorithms.quicksort(arr, low, pivot_index - 1)
            DivideAndConquerAlgorithms.quicksort(arr, pivot_index + 1, high)

    @staticmethod
    def partition(arr, low, high):
        pivot = arr[high]
        i = low - 1
        for j in range(low, high):
            if arr[j] < pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
        arr[i + 1], arr[high] = arr[high], arr[i + 1]
        return i + 1

    @staticmethod
    def closest_pair_1d(arr):
        arr.sort()
        min_diff = float('inf')
        for i in range(1, len(arr)):
            min_diff = min(min_diff, arr[i] - arr[i - 1])
        return min_diff

    @staticmethod
    def karatsuba_multiply(x, y):
        if x < 10 or y < 10:
            return x * y
        n = max(len(str(x)), len(str(y)))
        half = n // 2
        high1, low1 = divmod(x, 10 ** half)
        high2, low2 = divmod(y, 10 ** half)
        z0 = DivideAndConquerAlgorithms.karatsuba_multiply(low1, low2)
        z1 = DivideAndConquerAlgorithms.karatsuba_multiply((low1 + high1), (low2 + high2))
        z2 = DivideAndConquerAlgorithms.karatsuba_multiply(high1, high2)
        return (z2 * 10 ** (2 * half)) + ((z1 - z2 - z0) * 10 ** half) + z0

class InputGenerator:
    @staticmethod
    def generate_closest_pair_data(num_points):
        points = np.random.uniform(0, 1000, num_points)
        points.sort()
        return points

    @staticmethod
    def generate_integer_multiplication_data(num_digits):
        num1 = random.randint(10 ** (num_digits - 1), 10 ** num_digits - 1)
        num2 = random.randint(10 ** (num_digits - 1), 10 ** num_digits - 1)
        return num1, num2

def main():
    st.title("Divide and Conquer Algorithms")
    
    with st.sidebar:
        algo_options = {
            "Find Max/Min": "Find maximum and minimum values",
            "Exponentiation": "Calculate power using divide and conquer",
            "Count Inversions": "Count number of inversions in array",
            "Quicksort": "Sort array using quicksort",
            "Closest Pair 1D": "Find closest pair in 1D array",
            "Karatsuba Multiplication": "Multiply large integers"
        }
        selected_algo = st.selectbox("Select Algorithm", list(algo_options.keys()), format_func=lambda x: f"{x}: {algo_options[x]}")
        st.markdown(AlgorithmExplanations.get_explanation(selected_algo))
    
    tab1, tab2 = st.tabs(["Run Algorithm", "Generate Test Data"])
    
    with tab1:
        st.header("Run Algorithm")
        algo = DivideAndConquerAlgorithms()
        
        if selected_algo == "Exponentiation":
            st.subheader("Input for Exponentiation")
            base = st.number_input("Enter base:", value=2.0)
            exp = st.number_input("Enter exponent:", value=3, min_value=0)
            if st.button("Calculate Power"):
                try:
                    start_time = time.time()
                    result = algo.power(base, int(exp))
                    end_time = time.time()
                    st.success(f"Result: {result}")
                    st.info(f"Execution time: {end_time - start_time:.6f} seconds")
                except Exception as e:
                    st.error(f"Error during computation: {str(e)}")
        else:
            uploaded_file = st.file_uploader("Upload input file", type=["txt", "csv"])
            
            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        data = pd.read_csv(uploaded_file)
                        arr = data.values.flatten()
                    else:
                        content = uploaded_file.read().decode()
                        arr = [float(x) for x in content.split()]
                    st.write("Input data:", arr)

                    start_time = time.time()
                    
                    if selected_algo == "Find Max/Min":
                        max_val, min_val = algo.find_max_min(arr, 0, len(arr) - 1)
                        st.success(f"Maximum: {max_val}, Minimum: {min_val}")
                    
                    elif selected_algo == "Count Inversions":
                        inv_count = algo.count_inversions(arr.copy())
                        st.success(f"Number of inversions: {inv_count}")
                    
                    elif selected_algo == "Quicksort":
                        sorted_arr = arr.copy()
                        algo.quicksort(sorted_arr, 0, len(sorted_arr) - 1)
                        st.success("Quicksort completed successfully!")
                        st.write("Sorted array:", sorted_arr)
                    
                    elif selected_algo == "Closest Pair 1D":
                        min_diff = algo.closest_pair_1d(arr.copy())
                        st.success(f"Minimum difference: {min_diff}")
                    
                    elif selected_algo == "Karatsuba Multiplication":
                        if len(arr) >= 2:
                            result = algo.karatsuba_multiply(int(arr[0]), int(arr[1]))
                            st.success(f"Result: {result}")
                    
                    end_time = time.time()
                    st.info(f"Execution time: {end_time - start_time:.6f} seconds")
                    
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
            else:
                st.info("Please upload a file to run the selected algorithm (except Exponentiation).")
    
    with tab2:
        st.header("Generate Test Data")
        num_samples = st.number_input("Number of samples to generate:", min_value=1, value=10)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Generate for Closest Pair 1D"):
                data = InputGenerator.generate_closest_pair_data(num_samples)
                random.shuffle(data)
                closest_pair_file = "closest_pair_data.txt"
                with open(closest_pair_file, "w") as file:
                    file.write("\n".join(map(str, data)))
                st.download_button(
                    label="Download Closest Pair Data",
                    data=open(closest_pair_file, "r").read(),
                    file_name=closest_pair_file,
                    mime="text/plain"
                )
                st.write(data)
                
        with col2:
            num_digits = st.number_input("Number of digits for each number:", min_value=1, value=5)
            if st.button("Generate for Karatsuba Multiplication"):
                num1, num2 = InputGenerator.generate_integer_multiplication_data(num_digits)
                st.write(f"Generated numbers: {num1}, {num2}")
                karatsuba_file = "karatsuba_data.txt"
                with open(karatsuba_file, "w") as file:
                    file.write(f"{num1}\n{num2}")
                st.download_button(
                    label="Download Karatsuba Data",
                    data=open(karatsuba_file, "r").read(),
                    file_name=karatsuba_file,
                    mime="text/plain"
                )

if __name__ == "__main__":
    main()

