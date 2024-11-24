# Divide and Conquer Algorithms with Streamlit

This repository contains a Streamlit-based application to explore and run several divide-and-conquer algorithms. It includes detailed explanations, functionality to run algorithms with user-provided data, and the ability to generate test data for selected algorithms.

---

## Features

- **Interactive Interface**: Select and run algorithms directly in the Streamlit app.
- **Detailed Explanations**: Each algorithm comes with a description of its working, time complexity, and applications.
- **File Upload Support**: Upload input files (`txt` or `csv`) to test algorithms.
- **Test Data Generation**: Generate random test data for specific algorithms and download it as a file.
- **Execution Timing**: View execution time for each algorithm.

---

## Algorithms Included

1. **Find Max/Min**: Efficiently find the maximum and minimum values in an array using divide-and-conquer.
2. **Exponentiation**: Calculate powers using the fast exponentiation (exponentiation by squaring) technique.
3. **Count Inversions**: Count the number of inversions in an array using a modified merge sort.
4. **Quicksort**: Sort an array efficiently using the quicksort algorithm.
5. **Closest Pair in 1D**: Find the closest pair of numbers in a 1D array.
6. **Karatsuba Multiplication**: Multiply two large integers using the Karatsuba algorithm.

---

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/Pineapple4623/DivideAndConquer
    ```
2. Navigate to the project directory:
    ```bash
    cd DivideAndConquer
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4. Run the Streamlit application:
    ```bash
    python -m streamlit run app.py
    ```

---

## How to Use

### 1. Select Algorithm:
- Use the dropdown in the sidebar to select the algorithm you want to explore.
- Read the explanation in the sidebar to understand the algorithm.

### 2. Run Algorithm:
- **Exponentiation**: Input the base and exponent directly in the app.
- **Other Algorithms**: Upload a file containing the input data (`.txt` or `.csv`).

### 3. Generate Test Data:
- Navigate to the **Generate Test Data** tab.
- Generate random data for algorithms like Closest Pair or Karatsuba Multiplication.
- Download the generated data for further use.

---

## File Format for Input

- **Text Files (`.txt`)**:
  - A list of space-separated or newline-separated numbers.
  - Example for Closest Pair: `12.3 45.6 78.9`.

- **CSV Files (`.csv`)**:
  - Single-column data where each row represents a number.

---

## Example Usage

### Running Exponentiation
- **Input**: Base = `2`, Exponent = `8`.
- **Output**: `256`.

### Running Count Inversions
- **Input**: `[2, 4, 1, 3, 5]`.
- **Output**: `3 inversions`.

---

## Dependencies

- `streamlit`: For building the web application.
- `numpy`: For numerical operations.
- `pandas`: For processing CSV input files.

---

## Live Demo

Check out the hosted application here: [Divide and Conquer Streamlit App](https://areeb-divideandconquer.streamlit.app/)

---

## Contribution

Contributions are welcome! If you have ideas for new features or algorithms, feel free to open an issue or submit a pull request.

1. Fork the repository.
2. Create a new branch: 
    ```bash
    git checkout -b feature-name
    ```
3. Commit your changes: 
    ```bash
    git commit -m 'Add some feature'
    ```
4. Push to the branch: 
    ```bash
    git push origin feature-name
    ```
5. Open a pull request.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
