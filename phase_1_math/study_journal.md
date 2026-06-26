### Monday 5/25 logs

Watched 3b1b "Essence of Linear Algebra" Ep1 and Ep2.

Implemented vector class and basic arithmetic operations like vector addition, subtraction, scalar multiplication, and dot product.

Also included magnitude in there, and a shape method to work a bit like numpy.

**Question:** How does 3B1B's geometric view differ from what you thought about vectors?

- Honestly doesn't change my view too much but I did find it interesting to think of vectors as direction of movement, kind of plays more into the physics interpretation.

### Tuesday 5/26 logs


## Week 2

### Monday 6/15

Made the silly mistake of setting the inverse as the transpose :(

Question: When does a matrix have no inverse? What happens geometrically?

Answer: A matrix has no inverse when it's determinant is zero. This happens because when a matrix's determinant is zero, it is a lower dimension, so geometrically, inverting the matrix in a high-dimensional space is not feasible. An example for this would be if we try to invert a cube in a 3D space. If the determinant is non-zero than the cube, regardless of any linear transformation, can be inverted along the $\hat{i}, \hat{j}, \hat{k}$ axes. However, if the determinant is zero, than the cube could be a plane in a 3D space, which is not invertible as there is a lose of information.

What I mean by *loss of information* is that there are multiple distinct points of the original cube that gets mapped to the exact same location when it is flattened to a 2D plane.

### Tuesday 6/16

This was a touch one to implement!

Question: How does elimination relate to row reduction you see in textbooks?

Answer: This is basically algebraic elimination but when doing row reduction, we just extract the coefficients only and work with that.

### Wednesday 6/17

didn't have to change too much for this from the gaussian elimination code, just needed to do a full column sweep for zeroing out the upper and lower triangle of the matrix.

Question: What is the computational cost of inversion?

Answer: for my implementation, it is a time complexity of $O(n^3)$

### Thursday 6/18

this wasn't so bad but I didn't really do everything from scratch, ended up using numpy's determinant function and matrix rank function for this.

Question: When would you use Cramer's rule vs elimination in practice?

Answer: Gaussian elimination is a much faster operation than Cramer's rule! So I would use gaussian elimination when execution time is vital. Most likely, in cases where the matrices of fairly large and of higher dimensions.

### Friday 6/19

Had to look up formula for NxN matrix determinants and inverses: [here](https://www.cliffsnotes.com/study-guides/algebra/linear-algebra/the-determinant/laplace-expansions-for-the-determinant)

Then I had to brush up a bit on recursion to get the det() function working properly.

I keep making the same error with the shallow copys: `matrix.copy()` 

This is a problem especially if I am trying to mutate a matrix in place, it will just reread or reswap my previous mutations so I should always default to creating a new matrix in place rather than copy and mutate the matrix being fed in.

I didn't really account for the no-solution and infinite solution cases in this implementation since I am being lazy but it is in my day 4 which classifies based on matrix rank. Here the only check I ensure is that the det is non-zero since my computations are inverse-driven.

My grade for myself is like a 60% honestly... Made some coding mistakes and had to look up the generalized equations for determinates and inverses since I was only focused on the 2x2 and 3x3 cases earlier in the week.

One thing I will point out is that gaussian eliminations is clearly better for runtime optimization... This cofactor computation in my from-scratch implementation is O(n!) and elimination is O(n^3)

### Saturday 6/20

Optional day of norm funcs and similarity funcs

So far I feel comfortable with l1, l2, l_inf. Will try cosine sim without any reference...

Ok important distinction, inner product and cosine similarity can both tell the relation between two vectors. However, inner product tells us the angle and magnitude between two vectors, while cosine sim will only express the relative angle between them.

formula for cos sim: sim(v1, v2) = (v1 \dot v2) / (||v1|| * ||v2||)

### Sunday 6/21

catching up since I am a week behind

## Week 3

### Monday 6/22

Eigenvectors are vectors that remain on the span during a transformation.

Eigenvalues are the values by which a eigenvector is stretched or shrinked during a transformation (a scalar value).

Eigenvectors during a 3D rotation become the "axis of rotation" since they remain on their span during the transformation. The eigenvalue in the case of rotation would be 1 since they don't stretch or shrink anything.

This helps provide a faster way of what linear transformations do to a coordinate system. Rather than transform the entire matrix of coordinates, we can determine the eigenvectors and their eigenvalues to see what the outcome of the linear transformation is!

$$ A \cdot \vec{v} = \lambda \vec{v}$$

The matrix-vector multiplication of $A$ and $\vec{v}$ is the same as the scalar multiplication of the eigenvalue $\lambda$ and the $\vec{v}$

Finding the eigenvectors and eigenvalues comes down to finding the values of $\lambda$ and $\vec{v}$ that make the above expression true.

Rewriting the expression: $(A - \lambda \cdot I )\vec{v} = \vec{0}$

Where $(A - \lambda \cdot I)$ would look something like:

$$ 
\begin{bmatrix}
a - \lambda & b & c \\
e & f - \lambda & g \\
h & i & j - \lambda
\end{bmatrix}
$$

This expression can be true in two cases, if $\vec{v} = \vec{0}$ or $\text{det}(A - \lambda \cdot I) = 0$

If there are no real number solutions to $\lambda$ then there are not eigenvectors of the linear transformation! an example of this would be a 90 deg rotation on a 2D plane. 

Eigenbasis can be when basis vectors happen to be the eigenvectors as well.

Ok my first attempt to implement the power iteration method was flawed because I did not reset the value of $\vec{v}_i$ once I normalized it to a unit vector. This messed with the convergence to the dominant vector due to integer overflow. 

The fix was setting the type of the matrix $A$ to a float and then normalizing the vector after each iteration, then setting it to value of the next iterations $\vec{v}_i$. I also don't need to apply the power of A to v each time, just A alone works.

I am also not really handling edge cases here like when the eigenvalues are imaginary.

If we check the diff of the eigenvalue for the old vector and the new vector, this gives a good idea of when the power iter converges.

A better, more robust check is to see if the values of v and lambda satify the residual $\| A \vec{v} - \lambda \vec{v}\|$. This is a stronger check for when to stop the power iteration process and that we identified a dominant eigenvector and eigenvalue.

**Question:** What does an eigenvector "mean" geometrically? Why does ML care?

**Answer:** geometrically, the eigenvector tells us which points in the vector space remain on their span after a linear transformation. This can help pinpoint axes for which a transformation does not alter and rather rotates or transforms other vectors around it. This is useful in ML because it could identify vectors that remain consistent or are a "pattern" in a multi-dimensional space of data.

### Tuesday 6/23

For determinants, we need to first test a matrix for invertibility:

$$AA^{-1} = I$$

$$\text{det}(A) = \begin{vmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{vmatrix} = a_{11}a_{22} - a_{12}a_{21}$$

For a square matrix $T$ that is a *upper-triangular matrix* if $T_{ij} = 0$ for $i > j$, the determinant is the product of the diagonal elements:

$$\text{det}(T) = \prod_{i=1}^{n}T_{ii}$$

For the determinant of a $n \times n$ matrix we can use a algorithm for the $n > 3$ cases:

(*Laplace Expansion*)

1. Expansion along column $j$

$$\text{det}(A) = \sum_{k=1}^{n}(-1)^{k+j}a_{kj}\text{det}(A_{k,j})$$

2. Expansion along row $i$

$$\text{det}(A) = \sum_{k=1}^{n}(-1)^{k+i}a_{ik}\text{det}(A_{i,k})$$

here, $A_{k,j} \in \mathbb{R}^{(n - 1)\times (n-1)}$ is the *submatrix* of $A$ that we get when deleting row $k$ and column $j$.

Jumping ahed a bit, Eigendecomposition follows the following expression:

$$AP = PD$$

Where $A$ is our matrix, $P$ is a matrix of eigenvectors $\vec{p_1}, \vec{p_2}, \ldots, \vec{p_n}$ for matrix $A$, and $D$ is a diagonal matrix of eigenvalues $\lambda_1, \lambda_2, \ldots, \lambda_n$ for matrix $A$. 

Therefore, a *square matrix* $A \in \mathbb{R}^{n \times n}$ can be factored into

$$A = PDP^{-1}$$

where $P \in \mathbb{R}^{n \times n}$ and $D$ is a diagonal matrix whose diagonal entries are the eigenvalues of $A$, if and only if the eigenvectors of $A$ form a basis by $\mathbb{R}^{n}$.

This makes solving for the eigendecomposition fairly straight forward if we are given the eigenvalues and eigenvectors of a matrix. 

**Question:** What types of matrices are diagonalizable?

**Answer:** Matrices with a eigenbasis that falls in $\mathbb{R}^{n}$. Additionally matrices that are real symmetric matrices are always diagonalizable ($A = A^{T}$).

### Wednesday 6/24

Singualar Value Decomposition (SVD) is a way to factor in linear algebra that decomposes a matrix into three other matrices. This allows for a way to represent data in terms of its singualar values.

Mathematically, the SVD of a martix $A$ (of size $m \times n$) is represented by:

$$A = U\Sigma V^{T}$$

Where:
- $U$ is an $m \times m$ orthogonal martix whose columns are the left singular vectors of $A$.
- $\Sigma$ is a diagonal $m \times n$ matrix containing the singular values of $A$ is descending order.
- $V^{T}$ is the transpose of an $n \times n$ orthogonal matrix, where the columns are the right singular vectors of $A$.

Easiest way to understand this is through an example of calculating it. Let $A = \begin{bmatrix} 3 & 2 & 2 \\ 2 & 3 & -2 \end{bmatrix}$

We first calculate $AA^{T}$:

$$A = \begin{bmatrix} 3 & 2 & 2 \\ 2 & 3 & -2 \end{bmatrix}$$
$$A^{T} = \begin{bmatrix} 3 & 2 \\ 3 & 2 \\ -2 & 2 \end{bmatrix}$$
$$ A \cdot A^{T} = \begin{bmatrix} 3 & 2 & 2 \\ 2 & 3 & -2 \end{bmatrix} \cdot \begin{bmatrix} 3 & 2 \\ 3 & 2 \\ -2 & 2 \end{bmatrix} = \begin{bmatrix} 17 & 8 \\ 8 & 17 \end{bmatrix}$$

Then we calculate the eigenvalues of $AA^{T}$:

$$\text{det}(AA^{T} - \lambda I) = 0$$
$$\text{det} \begin{bmatrix} 17 - \lambda & 8 \\ 8 & 17 - \lambda \end{bmatrix} = 0$$
$$(\lambda - 25)(\lambda - 9) = 0$$

The values for $\lambda_1 = 25$ and $\lambda_2 = 9$. These eigenvalues correspond to the singular values $\sigma_{1} = 5$ and $\sigma_2 = 3$ since the singular values are the square roots of the eigenvalues.

Next we find the *Right Singular Vectors* (Eigenvectors of $A^{T}A$):

**For $\lambda_1 = 25$**: solve $(A^{T}A - 25 I) = \begin{bmatrix} -12 & 12 & 2 \\ 12 & -12 & -2 \\ 2 & -2 & -17 \end{bmatrix}$

we row-reduce this matrix to: $\begin{bmatrix} 1 & -1 & 0 \\ 0 & 0 & 1 \\ 0 & 0 & 0 \end{bmatrix}$

The eigenvector corresponding to $\lambda = 25$ is:

$$v_{1} = \begin{bmatrix} \frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} \\ 0 \end{bmatrix}$$

**For $\lambda = 9$:** Solve $(A^{T}A - 9 I)v = 0$

The eigenvector corresponding to $\lambda = 9$ is $v_2 = \begin{bmatrix} \frac{1}{\sqrt{18}} \\ \frac{-1}{\sqrt{18}} \\ \frac{4}{\sqrt{18}} \end{bmatrix}$

for the third eigenvector $v_3$: Since $v_3$ must be perpendicular to $v_1$ and $v_2$, we solve the system $v_{1}^{T}v_{2} = 0$ and $v_{2}^{T}v_{3} = 0$, leading to:

$$v_{3} = \begin{bmatrix} \frac{2}{3} \\ \frac{-2}{3} \\ \frac{-1}{3}  \end{bmatrix}$$

Now we can finally compute the *Left Singular Vectors* (Matrix $u$):

we use the formula $u_{i} = \frac{1}{\sigma_{i}} Av_{i}$. This results in

$$U = \begin{bmatrix} \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} & \frac{-1}{\sqrt{2}} \end{bmatrix}$$

This leaves us with the final SVD equation for matrix $A$:

$$U = \begin{bmatrix} \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} & \frac{-1}{\sqrt{2}} \end{bmatrix}$$
$$\Sigma = \begin{bmatrix} 5 & 0 & 0 \\ 0 & 3 & 0 \end{bmatrix}$$
$$V^{T} = \begin{bmatrix} \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} & 0 \\ \frac{1}{\sqrt{18}} & \frac{-1}{\sqrt{18}} & \frac{4}{\sqrt{18}} \\ \frac{2}{3} & \frac{-2}{3} & \frac{-1}{3} \end{bmatrix}$$

Thus, the SVD matrix of $A$ is: 

$$ A = \begin{bmatrix} \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} & \frac{-1}{\sqrt{2}} \end{bmatrix} \begin{bmatrix} 5 & 0 & 0 \\ 0 & 3 & 0 \end{bmatrix} \begin{bmatrix} \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} & 0 \\ \frac{1}{\sqrt{18}} & \frac{-1}{\sqrt{18}} & \frac{4}{\sqrt{18}} \\ \frac{2}{3} & \frac{-2}{3} & \frac{-1}{3} \end{bmatrix} $$

To sum this up differently:

- Step 1 & 2: Right-singular vectors as the eigenbasis of $A^{T}A$. We compute the singular values and right-singular vectors $v_{j}$ through the eigenvalue decomposition of $A^{T}A$, which is given as $A^{T}A = PDP^{T}$ where the columns of $P$ is equal to the columns of $V$ and the singular values $\sigma_{i}$ is the non-zero values of matrix $D$ converted to match the size of matrix $A$.
- Step 3: Left-singular vectors as the normalized image of the right-singular vectors. $u_{i} = \frac{1}{\sigma_{i}}Av_{i}$ and $U = \begin{bmatrix} u_{1}, u_{2}, \ldots, u_{i} \end{bmatrix}$.


After going through this example, it makes sense on how to do this on paper. But there is still a disconnect for me on how to tie this into a programmatic approach...

Key thing to remember is that the function `np.linalg.eig` returns a matrix of eigenvectors as **columns** not rows.

**Question:** How does SVD relate to eigendecomposition? When do you use which?

**Answer:** SVD and eigendecomp are both matrix factorization techniques but SVD is a direct generalization of eigendecomposition. Eigendecomposition only really works on square matrices but SVD can work on rectangular matrices as well. 

### Thursday 6/25

Going over principal component analysis today...

PCA is a hierarchical coordinate system (based on data) that gets the directions in your data that capture the maximum amount of variance in your data.

We have a dataset $X$ where each row is a measurement from a single experiment (a data sample), and each column is features of the data. We want to capture the dominant combination of features that describe as much of the data as possible. We can do this with SVD with some extra steps.

1. Compute the mean row: $\bar{x} = \frac{1}{n}\sum_{j=1}^{n}x_{j}$
2. Compute the average matrix: we take the vector of averages and multiply it with a vector of 1's
$$\bar{X} = \begin{bmatrix}1 \\ 1 \\ \vdots \\ 1_n\end{bmatrix} \cdot \begin{bmatrix}\bar{x}_1 & \bar{x}_2 & \cdots \bar{x}_i\end{bmatrix} \text{for features }i$$
3. Subtract the mean (mean-centered data): $B = X - \bar{X}$
4. Covariance Matrix of the rows of $B$. (Also known as the *Correlation Matrix* from the SVD $X^{T}X$): $C = B^{T}B$
5. Compute the Eigendecomposition of $C$: 
$$v_{i}B^{T}Bv_{i}$$
$$CV = VD$$
where $V$ is the eigenvectors of $C$ and $D$ is a diagonalized matrix of the eigenvalues of $C$. 

From this, we can compute the *Principal Components* $T$ by taking the SVD of $B$, such that $T = BV = U\Sigma$ and $B = U\Sigma V^{T}$. Here the eigenvectors matrix $V$ are also called the *loadings*.

> This can be a shortcut for our coding implementation! By computing the SVD of our mean-centered data $B$, we can get the principal components $T$.

We can also express the variance in the data by our eigenvalues. $\lambda = \sigma^{2}$. $\frac{\sum_{k=1}^{r} \lambda_{k}}{\sum_{k=1}^{n} \lambda_n}$ for $r$ modes => this gives us how many modes can be used to express the amount of variance captured by my first $r$ eigenvalues to express the variance in the data $X$. This can help us decide how many principal components we would want to keep to express x% of the variance in our data.

### Friday 6/25

Accidentally did day 5 for day 4 ... whoops!!!



