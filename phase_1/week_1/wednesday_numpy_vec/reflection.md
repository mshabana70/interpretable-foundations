## Question

Where does numpy's speed come from?

```bash
Total execution time for NumPY class: 0.099165
Total execution time for Custom: 0.110340
```

- Numpy uses external libraries like BLAS and LAPACK to do most of it's computation for linalg operations. Also, our custom vector code is not optimized to avoid python's type checks before every operation, making it slower than the numpy equivalent.