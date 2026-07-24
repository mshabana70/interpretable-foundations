### Monday 5/25 logs

Watched 3b1b "Essence of Linear Algebra" Ep1 and Ep2.

Implemented vector class and basic arithmetic operations like vector addition, subtraction, scalar multiplication, and dot product.

Also included magnitude in there, and a shape method to work a bit like numpy.

**Question:** How does 3B1B's geometric view differ from what you thought about vectors?

- Honestly doesn't change my view too much but I did find it interesting to think of vectors as direction of movement, kind of plays more into the physics interpretation.

### Tuesday 5/26 logs


## Week 2 - Determinants, Inverses & Solving Systems

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

## Week 3 - Eigenvalues, Eigenvectors & PCA

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

**Question:** Why does PCA use the covariance matrix?

**Answer:** It uses the covariance matrix because the goal of PCA is to find the directos of maximum variance in high-dim data. the Covariance matrix helps capture this mathematically.

### Friday 6/25

Accidentally did day 5 for day 4 ... whoops!!!

Cleaned up the graphs for day 5 and learned some matplotlib trickery.

**Question:** How much variance did your first 2 components capture?

**Answer:** Roughly 97% of the total data variance was captured by the first two principal components. Again, this can be calculated by $\frac{\sum_{k=1}^{r} \lambda_{k}}{\sum_{k=1}^{n} \lambda_n}$ where r is the number of components we want to use and the value returned is the variance they cover.

### Saturday 6/26

skipped saturday this week, will circle back to it inshallah.

### Sunday 6/28

did an extra problem on deep-ml.com for PCA implementation. Then did the 3sums leetcode problem to practice two pointer algorithms.

## Week 4 - Calculus: Derivatives & the Chain Rule

### Monday 6/29

We need to implement numerical differentiation as a func `numerical_derivative(f, x, h=1e-7)` that approximates $\frac{df}{dx}$. 

Our test conditions are $x^{2}$, $\sin{x}$, and $e^{x}$. We can compare against analytical derivatives to verify (which I'll probably just hard-code the analytical derivative for the test cases).

**Finite difference methods** calculate the derivative $f'(x)$ at a specific point using a small step size $h$ (the distance between points):

$$f'(x) \approx \frac{f(x+h) - f(x-h)}{2h}$$

This is the central difference and the most accurate approximation when it comes to finite difference methods. There is also the *forward difference* and *backward difference* respectively:

$$f'(x) = \frac{f(x + h) - f(x)}{h}$$

$$f'(x) = \frac{f(x) - f(x - h)}{h}$$

That was fairly straight forward to implement but one thing to note is I had to do a threshold equivalance for the numerical differentiation and the hardcoded derivative (since they are approximations). The threshold was diff < 1e-4

**Question:** Why does `h=1e-7` work better than `h=e-15`?

**Answer:** This has to do with how computer precision is calculated. If $h$ is too small of a step size, $f(x+h)$ and $f(x)$ or $f(x-h)$ become nearly identical. If we subtract these values in the numerator, this results in a loss of significant figures or a round-off error. This causes the approximation to blow out or flatline to zero. 

### Tuesday 6/30

A bit lost on how to do the OOD design of this...

Will try to start simple by creating an `Add` sub-class that evaluates, differentiates, and just has a string output when printed. 

Using a tree structure where each op has leaves. The ops pass the expressions on to the children to gather values from `feed_dict`. 

I used claude to generate the `test()` func because I was lazy ...

This program is complete! Need to review this throughout the week.

**Question:** How does the chain rule work as "multiplication of local stretching factors"?

**Answer:** Because the chain rule states that the derivative of a composite function is the product of the derivatives of its outer and inner functions. Visually, this is how much the inner function stretches the input space, multiplied by how much the outer function stretches the intermediate result. 

**Update:** I decided to try to handle the case where we differentiate a power expression whose exponent is a type `Variable`. This requires a different approach than if `Power().exponent` was a type `Constant`. Since we are doing differentiation of two functions $f(x)^{g(x)}$, we can define it in a new format using *log differentiation*:

$$u = f(x); v = g(x)$$
$$\frac{d}{dx}(u^{v}) = \frac{d}{dx}(e^{v\cdot ln(u)})$$

So we can set a condition in `Power().diff()` that checks the instance of our `self.exponent`. If it is a `Constant` type, we just apply the power rule. 

If `self.exponent` is anything other than a `Constant` type (since it is not a `Constant`, we just place it in an else block), then we apply the log differentiation rule.

Also, one note to add, I used Monday's numerical differentiation approximation to test how accurate the symbolic derivatives that we implemented today. 

### Wednesday 7/1

We need to compute the gradient of a multivariable function. We can do this with the *finite difference method* as we did for the univariable function on monday.

For the *central difference method* of a multivariable function:

$$\nabla f(\textbf{x}) = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \frac{\partial f}{\partial x_3} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{bmatrix}$$

$$\frac{\partial f}{\partial x_i} \approx \frac{f(\textbf{x} + he_{i}) - f(\textbf{x} - he_{i})}{2h}$$

Here $e_{i}$ is the standard unit vector for dim $i$.

**Question:** What is a gradient geometrically?

**Answer:** A gradient is a measure of steepness, incline, or decline a line. It indicate how much a line moves vertically for every unit it moves horizontally.

### Thursday 7/2

Going through the "Mathematics for Machine Learning" book for this task today. Specifically section 5.1-5.3.

Definition of the *Taylor Polynomial* of degree $n$ for $f : \mathbb{R} \to \mathbb{R}$ at $x_{0}$ is defined as:

$$T_{n}(x) := \sum_{k=0}^{n} \frac{f^{(k)}(x_{0})}{k!}(x - x_{0})^{k},$$

where $f^{(k)}(x_{0})$ is the $k$th derivative of $f$ at $x_{0}$ (which we assume exists in this case) and $\frac{f^{(k)}(x_{0})}{k!}$ are coefficients of the polynomial.

For a continuous function $f \in \mathcal{C}^{\infty}$ ($\mathcal{C}$ is the set of continuous differentiable functions), we can define the *Taylor Series* as:

$$T_{\infty}(x) := \sum_{k=0}^{\infty} \frac{f^{(k)}(x_{0})}{k!}(x - x_{0})^{k},$$

Taylor series is a special case of *power series*:

$$f(x) = \sum_{k=0}^{\infty}a_{k}(x - c)^{k}$$

where $a_{k}$ are coefficients and $c$ is a constant. 

Important differentiation rules to keep in mind:

- **Product Rule:** $(f(x)g(x))' = f'(x)g(x) + f(x)g'(x)$
- **Quotient Rule:** $\left( \frac{f(x)}{g(x)} \right)' = \frac{f'(x)g(x) - f(x)g'(x)}{(g(x))^{2}}$
- **Sum Rule:** $(f(x) + g(x))' = f'(x) + g'(x)$
- **Chain Rule:** $(g(f(x)))' = (g \circ f)'(x) = g'(f(x))f'(x)$

$g \circ f$ means function composition $x \mapsto f(x) \mapsto g(f(x))$.

As covered yesterday, when $f$ depends on one or more variables $x \in \mathbb{R}^{n}$ (like $f(\textbf{x}) = f(x_{1}, x_{2})$). The generalization of the derivative to functions of several variables is the *gradient*.

We find the gradient of the function $f$ with respect to $x$ by *varying one variable at a time* and keeping the others constant. The gradient is then the collection of these *partial derivatives*

For a function $f : \mathbb{R}^{n} \to \mathbb{R}$, $x \mapsto f(\textbf{x})$, $\textbf{x} \in \mathbb{R}^{n}$ of $n$ variables $x_{1}, \ldots, x_{n}$ we define the *partial derivatives* as

$$
\begin{align*}
\frac{\partial f}{\partial x_{1}} &= \lim_{h \to 0}\frac{f(x_{1} + h, x_{2}, \ldots, x_{n}) - f(\textbf{x})}{h} \\
\vdots \\
\frac{\partial f}{\partial x_{n}} &= \lim_{h \to 0}\frac{f(x_{1},\ldots,x_{n-1},x_{n} + h) - f(\textbf{x})}{h}
\end{align*}
$$

an collect them in the row vector
$$
\nabla_{\textbf{x}}f = \text{grad}f = \frac{df}{d\textbf{x}} = \begin{bmatrix} \frac{\partial f(\textbf{x})}{\partial x_{1}} &  \frac{\partial f(\textbf{x})}{\partial x_{2}} & \ldots & \frac{\partial f(\textbf{x})}{\partial x_{n}}\end{bmatrix} \in \mathbb{R}^{1 \times n}
$$

here $n$ is the number of variables and 1 is the dimension of the image/codomain of $f$ (the function value dimension of $f(\textbf{x})$)

Vector $\textbf{x}$ is a row vector, the same row vector we used in the "$\nabla_{\textbf{x}} f$" formula above. This is the *gradient of $f$*, also known as the *Jacobian*. 

> NOTE: This definition of the Jacobian is a special case of the general Jacobian.

Basic rules of partial differentiation:

- **Product Rule:** $\frac{\partial}{\partial \textbf{x}}(f(\textbf{x})g(\textbf{x})) = \frac{\partial f}{\partial \textbf{x}}g(\textbf{x}) + f(\textbf{x})\frac{\partial g}{\partial \textbf{x}}$
- **Sum Rule:** $\frac{\partial}{\partial \textbf{x}}(f(\textbf{x}) + g(\textbf{x})) = \frac{\partial f}{\partial \textbf{x}} + \frac{\partial f}{\partial \textbf{x}}$
- **Chain Rule:** $\frac{\partial}{\partial \textbf{x}}(g(f(\textbf{x}))) = \frac{\partial}{\partial \textbf{x}}(g \circ f)(\textbf{x}) = \frac{\partial g}{\partial f}\frac{\partial f}{\partial \textbf{x}}$

Let's generalize this to vector-valued functions $\textbf{f} : \mathbb{R}^{n} \to \mathbb{R}^{m}$ where $n \geq 1$ and $m > 1$. For $\textbf{f} : \mathbb{R}^{n} \to \mathbb{R}^{m}$ and a vector $\textbf{x} = [x_{1},\ldots, x_{n}]^{\top} \in \mathbb{R}^{n}$, the corresponding vector of function values is 

$$\textbf{f}(\textbf{x}) = \begin{bmatrix} f_{1}(\textbf{x}) \\ \vdots \\ f_{m}(\textbf{x})\end{bmatrix} \in \mathbb{R}^{m}$$

"Writing the vector-valued function this way allows us to view a vector-valued function $\textbf{f} : \mathbb{R}^{n} \to \mathbb{R}^{m}$ as a vector of functions $[ f_1, \ldots , f_m]^{\top}$, $f_{i} : \mathbb{R}^{n} \to \mathbb{R}$ that map onto $\mathbb{R}$."

The partial derivative of a vector-valued function $\textbf{f} : \mathbb{R}^{n} \to \mathbb{R}^{m}$ with respect to $x_{i} \in \mathbb{R}, i = 1, \ldots, n$ is given as a vector

$$
\frac{\partial \textbf{f}}{\partial x_{i}} = \begin{bmatrix} \frac{\partial f_1}{\partial x_{i}} \\ \vdots \\ \frac{\partial f_m}{\partial x_{i}}\end{bmatrix} = \begin{bmatrix}\lim_{h \to 0} \frac{f_1(x_1, \ldots, x_{i-1}, x_{i} + h, x_{i+1}, \ldots, x_{n}) - f_1(\textbf{x})}{h} \\ \vdots \\ \lim_{h \to 0}\frac{f_m(x_1, \ldots, x_{i-1}, x_{i} + h, x_{i+1}, \ldots, x_{n}) - f_m(\textbf{x})}{h}\end{bmatrix} \in \mathbb{R}^{m}
$$

From our earlier definition of $\nabla_{\textbf{x}}f$, we know that the gradient of $\textbf{f}$ with respect to a vector is the row vector of the partial derivatives. In our above equation, every partial deriative $\frac{\partial \textbf{f}}{\partial x_{i}}$ is itself a column vector. Therefore:

$$
\begin{align*}
\frac{d\textbf{f}}{d\textbf{x}} &= \begin{bmatrix} \boxed{\frac{\partial \textbf{f}(\textbf{x})}{\partial x_{1}}} & \cdots & \boxed{\frac{\partial \textbf{f}(\textbf{x})}{\partial x_{n}}}\end{bmatrix} \\
&= \begin{bmatrix} \boxed{\begin{matrix} \frac{\partial f_1(\textbf{x})}{\partial x_{1}} \\ \vdots \\ \frac{\partial f_m(\textbf{x})}{\partial x_{1}}\end{matrix}} & \cdots & \boxed{\begin{matrix} \frac{\partial f_1(\textbf{x})}{\partial x_{n}} \\ \vdots \\ \frac{\partial f_m(\textbf{x})}{\partial x_{n}}\end{matrix}} \end{bmatrix} \in \mathbb{R}^{m \times n}
\end{align*}
$$

NOW for the **Jacobian**: The collection of all first-order partial derivatives of a vector-valued function $\textbf{f}: \mathbb{R}^{n} \to \mathbb{R}^{m}$. The Jacobian $\textbf{J}$ is an $m \times n$ matrix, which we can define as:

$$
\begin{align*}

\textbf{J} &= \nabla_{\textbf{x}}\textbf{f} = \frac{d\textbf{f(\textbf{x})}}{d\textbf{x}} = \begin{bmatrix} \frac{\partial \textbf{f}(\textbf{x})}{\partial x_{1}} & \cdots & \frac{\partial \textbf{f}(\textbf{x})}{\partial x_{n}}\end{bmatrix} \\

&= \begin{bmatrix} \frac{\partial f_1(\textbf{x})}{\partial x_{1}} & \cdots & \frac{\partial f_1(\textbf{x})}{\partial x_{n}} \\ \vdots &  & \vdots \\ \frac{\partial f_m(\textbf{x})}{\partial x_{1}} & \cdots &  \frac{\partial f_m(\textbf{x})}{\partial x_{n}}\end{bmatrix}, \\ 

\textbf{x} &= \begin{bmatrix} x_1 \\ \vdots \\ x_{n} \end{bmatrix}, \space\space\space\space J(i, j) = \frac{\partial f_{i}}{\partial x_{j}}.

\end{align*}
$$

Little side note, numpy's `zeros_like()` takes the array you want to match and can define the datatype for the zeros array. `zeros()` needs a tuple of the shape of the zeros array you want to create. 

**Question:** When do you need the Jacobian vs just the gradient?

**Answer:** The jacobian is useful when differentiation a vector-valued function on a vector of variables. The gradient can be computed only when you are dealing with a function that maps the $\mathbb{R}^2 \to \mathbb{R}$.

### Friday 7/3

The algo for gradient descent:

$$x^{(k+1)} = x^{(k)} - \alpha \nabla_{x}f(x^{(k)}$$

Here, we start with an initial point for $x$ and set it to the value of $x^{(k)}$. Then, calculate the gradient of our function $f(x)$ with respect to $x^{(k)}$ and multiply it with a step size or *learning rate* to scale our update magnitude. This helps us avoid overshooting any minimum we descend to.

Once we calculate this gradient, we take the learning rate - gradient product and subtract it from the value of $x^{(k)}$ (our current position on the function). This gives us a *descent* to a new point (descending because we subtract the gradient value, going down the slop of at the $x^{(k)}$ ). This gives us the new point $x^{(k+1)}$. Now we have moved from our original position on the graph and need to repeat this process by resetting the value of $x^{(k)}$ to $x^{(k+1)}$. 

For a stopping condition, I decided to measure the update distance between $x^{(k)}$ and $x^{(k+1)}$. This can inform us that whenever we are moving by a very small distance, we have reached a minimum of the function $f(x)$. This obviously doesn't tell us the difference on whether we found a local or global minimum so there is improvement to be had here.

**Question:** Did gradient descent find the global minimum? Why or why not?

**Answer:** Our example did find the global minimum, but that was because we chose a good point on the graph for our initial value of $x^{(k)}$. If we chose $x^{(k)} = [3.5, 5.0]$, we would fall into a local minimum farrrr from the global (just a local ridge due to $2\sin(xy)$). 

## Week 5 - Review Week

### Wednesday 7/8

Not too much to note this day. Just implemented Linear regression which wasn't too bad and redid gradient descent from memory with some look up on cost functions. 

One thing to note about least squares approximation:

You need to preprend a column of 1s to the feature matrix X so that you can actually do the $(X^{\top}X)^{-1}$ operation, as well as calculate a bias term from least squares.

Otherwise you will get a no-op error from numpy for doing the inverse of a scalar (since numpy does not tell the difference between a row and column when it comes to 1-D arrays).

Fixing this with `column_stack()` led to the two correct values for bias and weight.

Optional thing to do here: Standardize X using *z-score normalization* to improve the speed of GD convergence. Essential restructure X to be a mean of 0 and have all values of the X be in terms of standard deviations from the mean. 

## Week 5: Multivariable Calculus & Optimization 

### Monday 7/13

$$\nabla f(x,y) = \left[\ \frac{\partial f}{\partial x},\ \frac{\partial f}{\partial y}\ \right]\Bigg|_{(x,y)}$$

$$x \leftarrow x - \alpha,\frac{\partial f}{\partial x}, \qquad y \leftarrow y - \alpha,\frac{\partial f}{\partial y}$$

**Question:** How does learning rate affect convergence?

**Answer:** Extremely small LRs make convergence incredibly slow and inefficient. However, if the LR is too large then you could overshoot the global minimum search and oscillate forever, causing a failure in convergence.

### Tuesday 7/14

didn't do a damn thing today smh.

### Wednesday 7/15

Still didn't do a damn thing my goodness!!!

### Thursday 7/16

Going to start with catching up on day 2 tasks:

With vanilla GD, we go wherever the steepest descent is, regardless of whether or not it leads to the global minimum.

With momentum GD, we add a *velocity* to our descent, carrying us faster down the descent and also dampening whenever we hit a bump or a incline. 

With Nesterov GD, we still have this *velocity* from momentum but it is a bit smarter than that. We now **look ahead** to see where the momentum is going to take us in the descent. If we see that momentum is going to take us up an incline or flip from *descent* to *ascent*, we stop the descent and correct ourselves.

Let's put this in mathematical terms:

**Vanilla GD:** We are trying to minimize an objective function $f(\theta)$, where $\theta$ represents our parameters. At any step $t$, the gradient of the loss surface with respect to our parameters is $\nabla f(\theta_{t})$. Let $\eta$ be our learning rate. When we calculate our update, it is completely localized. We take the gradient at our current position and take a step:

$$\theta_{t+1} = \theta_{t} - \eta\nabla f(\theta_{t})$$

The issue with this approach is when our loss surface has high curvature (an ill-conditioned loss), the gradient ends up pointing across these curves rather than down it, leading to oscillation. Additionally, there is no way to avoid ending up in shallow valleys or local minimums. 

**Momentum GD:** To help address the oscillation problem, we add a velocity vector $v_{t}$ and a momentum coefficient $\beta$ (usually around 0.9). 

$$
\begin{align*}
v_{t+1} &= \beta v_{t} + \eta\nabla f(\theta_{t}) \\
\theta_{t+1} &= \theta_{t} - v_{t+1}
\end{align*}
$$

The step we take here ($- v_{t+1}$) is a vector addtion of two things:
1. Our accumulated history $\beta v_{t}$ (where our momentum is carrying you)
2. Our current gradient $\eta\nabla f(\theta_t)$ (the slope we are currently positioned on)

The new issue with this now is that momentum is blind and is just taking us wherever the gradient says to go, just at a greater speed now. There is no way to know if we head in the wrong direction or make a mistake (overshoot a minimum) until after the update since the gradient is only computed at the current position.

**Nesterov GD:** This is where we improve our approach AGAIN and address our momentum blindness. If we know that our momentum term $\beta v_{t}$ is going to drap our parameters forward regardless of what the current gradient says, why calculate the gradient at our current position? 

Instead, we should calculate the gradient at our *look-ahead position*: $\theta_{t} - \beta v_{t}$. This makes our update:

$$
\begin{align*}
v_{t+1} &= \beta v_{t} + \eta\nabla f(\theta_t - \beta v_{t}) \\
\theta_{t+1} &= \theta_t - v_{t+1}
\end{align*}
$$

The difference now is that instead of evaluating the gradient at $\theta_t$, we take a phantom step using *only* the momentum $\theta_t - \beta v_{t}$ and evaluate the gradient there. Once we get this look-ahead evaluation, we correct our momentum term based on that look-ahead position.

**Question:** Why does momentum help?

**Answer:** Momentum helps our descent from oscillating in a region of high curvature in the loss space. Instead of vanilla gradient descent, which will get stuck over-correcting when it jumps from one side of the valley to the next, momentum will allow us to slow down or dampen our descent until our current position gradient reaches a zero value. Additionally, it helps us speed past shallow valleys or local minimums, which can be more helpful in finding teh global minimum.

---

Moving on to Stochastic gradient descent.

The previous variants that we implemented where all about how we take a gradient step. For **Stochastic Gradient Descent (SGD)**, we are going to focus on what information we use to decide the direction of that gradient step.

To see how and why SGD works, we need to define a objective function first. Generally, our total loss is computed by $f(\theta)$ which is almost always the average of the individual losses calculated over our entire dataset of $N$ examples.

Let $L_{i}(\theta)$ be the loss for the $i$-th training example. The total objective function is:
$$
f(\theta) = \frac{1}{N}\sum_{i=1}^{N} L_{i}(\theta)
$$

**Full-batch SGD:**

To do vanilla gradient descent, we need the gradient of the total loss:

$$
\begin{align*}
\nabla f(\theta_{t}) &= \frac{1}{N} \sum_{i=1}^{N} \nabla L_{i}(\theta_{t}) \\
\theta_{t+1} &= \theta_{t} - \eta\nabla f(\theta_{t})
\end{align*}
$$

The issue with this approach is if $N$ is a billion or trillion, you must do a billion or trillion forward and backward passes just to update you parameters $\theta$ once. That is nuts so we gotta find a better way than this.

**Stochastic Approach: Single-batch**

The stochastic approach gives up on computing the exact gradient $\nabla f(\theta_t)$. Instead, at each step $t$, we draw a single index $i$ uniformly at random from our dataset $\{ 1, \ldots, N\}$. We then compute the gradient for *only that one example* and use it as our update:

$$\theta_{t+1} = \theta_{t} - \eta\nabla L_{i}(\theta_{t})$$

How does updating our gradient step based on a single, randomly chosen data point guarantee that we actually minimize the total loss function?

The core justification for SGD is that the stochastic gradient is an **unbiased estimater** of the true gradient. Because we draw the sample $i$ uniformly at random, the expected value of our noisy gradient is exactly equal to the true gradient:
$$
\mathbb{E}[\nabla L_{i}(\theta_t)] = \frac{1}{N}\sum_{i=1}^{N}\nabla L_{i}(\theta_t) = \nabla f(\theta_t)
$$

*In expectation*, SGD moves us in the exact same direction as Vanilla GD. However, because it is an expectation, an *individual step* introduces variance or noise. This can actually be helpful as a single-batch step can help the GD bounce out of local minimums or shallow valleys in the loss space.

**Stochastic Approach: Mini-batch**

Here, the algorithm is not too much different than our single-batch. However, instead of taking one example at random from our dataset, we can take 32 or 128 samples and compute the gradient on that batch of examples. Helps find a balance between slow full dataset gradients and noisy single data point gradients. Here is how the update would look using a mini-batch approach:

We define a batch size as $B$ (e.g., $B = 64$). At each step, we sample a subset of indices $S_{t} \subset \{1, \ldots, N\}$ where $|S_{t}| = B$. The update rule then becomes:

$$\theta_{t+1} = \theta_{t} - \eta\left( \frac{1}{B} \sum_{j \in S_{t}} \nabla L_{j}(\theta_{t}) \right)$$

$$L(w,b) = \frac{1}{N}\sum_{i=1}^{N}\big(\underbrace{wx_i + b}_{\hat y_i} - y_i\big)^2$$

**Question:** Why is SGD noiser but often faster?

**Answer:** SGD follows mini-batch estimates to descent to the minimum of a objective function. This means we do not need to compute the gradient of each data point in order to find the optimal steepest descent. We instead take a small sample of data points and compute the gradient of those to give us a estimate that we can trust to get us down the loss space faster. This is noisy though since it does not really take into account all data sample when updating our parameters. 

### Friday 7/17

Still catching up, working on Adam optimizer today:

We did SGD and Momentum GD the past couple of days, where we compute gradients either by sampling over a batch of data examples, or by adding a momentum term to the descent to help with dampening and a running accumulation of gradients as we descend to the minimum.

> NOTE: we did not make a note on Exponentially Weighted Moving Average (EWMA)... But when computing the momentum term $v_t$, $\frac{1}{1 - \beta}$ gives us the rough number of terms that the current momentum will be heavily dependent on. For example, when $\beta = 0.9$, approx the past 10 gradients will heavily influence the momentum at the current point. As $\beta$ approaches 1, more of the past values or gradients will influence the momentum term. Conversly, as we approach 0, less of the past values or gradients will influence the momentum term. **Bias Correction:** $v_{t}^{corrected} = \frac{v_t}{1 - \beta^{t}}$

**Adam Optimizer: Adaptive Moment Optimizer**

In order to understand Adam, we need to understand *RMSprop* first. *RMSprop* is all about balancing the magnitude of our gradient steps across all parameter axes. For example, usually in vanilla GD, our gradient updates for the weights tend to be more noisy than the bias term, causing our descent to take large steps across the weight axis, but smaller ones across the bias axis. 

We want to balance this out by taking scaled steps across each parameter of our gradient based on the magnitude of the gradient. The scaling of the steps is inverted, so a large gradient alters the learning rate to be much smaller ("This is a steep point on the surface, let me take a smaller step"). The inverse of this is true also, a small gradient means a larger gradient step ("This is a flat part of the surface, let's take a larger step"). 

$$
\begin{align*}
v_{t} &=  \beta v_{t-1} + (1 - \beta) * \nabla f(\theta_t)^{2} \\
\theta_{t + 1} &= \theta_t - \frac{\eta}{\sqrt{v_{t} + \epsilon}} \nabla f(\theta_{t})  
\end{align*}
$$

Here, $\epsilon$ is so we avoid a division by zero. It is also important to remember that *RMSprop* get's its own hyperparameter $\beta$, it is not the same value as the hyperparameter for the momentum term.

Now we combine both approaches. We take the benefits of momentum, which helps define which direction to accelerate in, and the benefits of RMSprop, which helps define how much we should step in that acceleration based on the scale of each gradient parameter. That is the Adam optimizer! Let's define it mathematically:

Let $m_{t}$ be our momentum, and $v_{t}$ be our RMSprop term (adaptive step). We compute Adam as follows:

$$
\begin{align*}
m_{t} &= \beta_{m}m_{t-1} + (1 - \beta_{m})\nabla f(\theta_t) \\
v_{t} &= \beta_{v}v_{t-1} + (1 - \beta_{v})\nabla f (\theta_t)^{2} \\
\hat{m_{t}} &= \frac{m_{t}}{1 - \beta_m^{t}} \\
\hat{v_{t}} &= \frac{v_{t}}{1 - \beta_v^{t}} \\
\theta_{t+1} &= \theta_{t} - \eta\frac{\hat{m_{t}}}{\sqrt{\hat{v_{t}}} + \epsilon}\\
\end{align*}
$$

**Question:** What problem does Adam solve that vanilla SGD doesn't?

**Answer:** In this scenario, nothing really. Adam introduces additional overhead when the problem is already well-behaved and isn't too noisy. Adam beats out SGD when the problem is high-dimensionality and ill-conditioned. There, Adam helps reduce the noise and scale its descent across high-variance parameter magnitudes.

---

Ok for the friday challenge we are going to implement an **optimizer showdown**. 

Need to implement a `train(optimizer_fn, X, y, ...)` function that will loop over 5 optimizers:

1. Vanilla GD
2. Momentum GD
3. Nesterov GD
4. Mini-batch SGD
5. Adam

The dataset will be a multi-feature linear regression with purposefully bad feature scaling (e.g., $x_1 \in [0, 1], x_2 \in [0, 10000]$) so $\kappa(X^{\top}X) $ is large.

The ground truth is the *normal equation* $\theta = (X^{\top}X)^{-1}X^{\top}y$ this will be our target plot

Then we need to run our test on unstandardized data, plot it, then rerun it on standardized data and compare plots. Should see the more optimizers fit the target data once the features are standardized. Adam should do well in both scenarios.


## Week 6 - Probability & Statistics for ML

### Monday 7/13

This week we need to implement MLE for a gaussian. Given $N$ samples, we need to compute the ML estimates of mean and variance from scratch. 

To verify, we compare with `np.mean` and `np.var`.

Probability is the value of a given event occuring based on the distribution that event is pulled from. For example, the probability of picking a sample that is 50 grams from a population with a normal distribution of mean = 50 and std = 2.5 would be 50%. 

Likelihood is a bit different as it measure the value of a given event as the point on the distribution that that event falls on. 

**In summary:** 

Probabilities are the areas under a fixed distribution ($Pr(\text{Data} | \text{Distribution})$). 

Likelihoods are the y-axis values for fixed data points with distributions that can be moved ($L(\text{Distribution} | \text{Data})$)

Probability is used for prediction, Likelihood is used for parameter estimation ($\theta$)

**Maximum Likelihood Estimation:**

The goal is to find the optimal way to fit a distribution to a given data sample. The reason you want to fit a distribution to your data is it can be easier to work with and it is also more general - it applies to every experiment of the same type.

We want to fit a distribution where the likelihood of observing all the data under a distribution with some given mean and std to be high, or at it's maximum. 

We essentially can try all of the possible values for the distribution mean and std where that likelihood is at it's maximum.

> NOTE: it is important to remember that we are looking for the mean of the *distribution*, not of the data. But it so happens when trying to fit a Gaussian distribution, these are the same. 


How do we compute the Maximum Likelihood Estimate (MLE)?

We start with the *probability density function (PDF)* of the Gaussian (Normal) Distribution:

$$pr(x | \mu, \sigma) = \frac{1}{\sqrt{2\pi\sigma^{2}}} e^{\frac{-(x - \mu)^{2}}{2\sigma^{2}}}$$

where the parameters for the PDF is $\mu$ (mean of distribution) and $\sigma$ (standard deviation of the distribution).

The likelihood can be computed with the same definition as above, except we are computing the parameters of the gaussian distribution $\mu$ and $\sigma$ given the data we observed $x$:

$$L(\mu, \sigma | x) = \frac{1}{\sqrt{2\pi\sigma^{2}}} e^{\frac{-(x - \mu)^{2}}{2\sigma^{2}}}$$

We can basically plug in values for $\mu$ and $\sigma$ (either one fixed) and compute the likelihood at each given point $x_{i}$ until we find the maximum likelihood.

Another way would be to:

If we solve for the maximum likelihood estimate for $\mu$, we treat $\sigma$ like a constant and then find where the slope of it's likelihood function is 0 (it's maximum "peak")


If we solve for the maximum likelihood estimate for $\sigma$, we treat $\mu$ like a constant and then find where the slope of it's likelihood function is 0 (it's maximum "peak")

Now if we want to solve for the maximum likelihood estimates given $n$ data samples $x_1, \ldots, x_n$, we take the product of all the likelihoods $L(\mu, \sigma | x_1) \times \ldots \times L(\mu, \sigma | x_n)$ which gives us the likelihood of the whole $n$ data samples and then the maximum of this result.

We could also just take the derivative of the Likelihood function with respect to $\mu$ and with respect to $\sigma$. We can also first take the log of the likelihood to make it easier to compute these derivatives:

$$
\begin{align*}
\log[L(\mu, \sigma | x_1, \ldots, x_n)] &= \log(\prod_{i=1}^{n} \frac{1}{\sqrt{2\pi\sigma^{2}}} e^{\frac{-(x_i - \mu)^{2}}{2\sigma^{2}}}) \\
&= \sum_{i=1}^{n} \log\left(\frac{1}{\sqrt{2\pi\sigma^{2}}} e^{\frac{-(x_i - \mu)^{2}}{2\sigma^{2}}}\right) \\
&= \sum_{i=1}^{n} \left( -\frac{1}{2}\log(2\pi) - \log(\sigma) - \frac{(x_{i} - \mu)^{2}}{2\sigma^{2}} \right) \\
&= -\frac{n}{2}\log(2\pi) - n\log(\sigma) - \sum_{i=1}^{n} \frac{(x_{i} - \mu)^{2}}{2\sigma^{2}}
\end{align*}
$$

Now we can take the derivative of this simplified log-likelihood function w.r.t $\mu$ and $\sigma$:

**With respect to $\mu$**:

$$
\begin{align*}
\frac{\partial}{\partial\mu}\log(L) &= \frac{\partial}{\partial\mu}\left( -\frac{n}{2}\log(2\pi) - n\log(\sigma) - \sum_{i=1}^{n} \frac{(x_{i} - \mu)^{2}}{2\sigma^{2}} \right) \\
&= 0 - 0 + \sum_{i=1}^{n}\frac{(x_{i} - \mu)}{\sigma^{2}} \\
&= \sum_{i=1}^{n}\frac{(x_{i} - \mu)}{\sigma^{2}} \\
&= \frac{1}{\sigma^{2}}[(x_1 + \ldots + x_{n}) - n\mu]
\end{align*}
$$

If we set the resulting derivative to 0, we can find the value of $\mu$ that maximizes the log likelihood:

$$
\begin{align*}
0 &= \frac{1}{\sigma^{2}}[(x_1 + \ldots + x_{n}) - n\mu] \\
0 &= \left( \sum_{i=1}^{n}x_{i} \right) - n\mu \\
n\mu &= \sum_{i=1}^{n}x_{i} \\
\mu &= \frac{1}{n} \sum_{i=1}^{n}x_{i}
\end{align*}
$$

**With respect to $\sigma$**:

$$
\begin{align*}
\frac{\partial}{\partial\sigma}\log(L) &= \frac{\partial}{\partial\sigma}\left( -\frac{n}{2}\log(2\pi) - n\log(\sigma) - \sum_{i=1}^{n} \frac{(x_{i} - \mu)^{2}}{2\sigma^{2}} \right) \\
&= 0 - \frac{n}{\sigma} + \sum_{i=1}^{n} \frac{(x_{i} - \mu)^{2}}{\sigma^{3}} \\
&= - \frac{n}{\sigma} + \frac{1}{\sigma^{3}}\sum_{i=1}^{n} (x_{i} - \mu)^{2}
\end{align*}
$$

If we set the resulting derivative to 0, we can find the value of $\sigma$ that maximizes the log likelihood:

$$
\begin{align*}
0 &= - \frac{n}{\sigma} + \frac{1}{\sigma^{3}}\sum_{i=1}^{n} (x_{i} - \mu)^{2} \\
\frac{n}{\sigma} &= \frac{1}{\sigma^{3}} \sum_{i=1}^{n} (x_{i} - \mu)^{2} \\
n\sigma^{2} &= \sum_{i=1}^{n} (x_{i} - \mu)^{2} \\
\sigma^{2} &= \frac{1}{n} \sum_{i=1}^{n} (x_{i} - \mu)^{2} \\
\sigma &= \sqrt{\frac{1}{n} \sum_{i=1}^{n} (x_{i} - \mu)^{2}}
\end{align*}
$$

**Question:** What's the difference between probability and likelihood?

**Answer:** Likelihood is really for determining the parameters for a target distribution given the data you have. While probability is about the how likely a data sample is to occur based on a known distribution that the data fits.

---

**Bayes Theorem:**

The definition of Bayes theorem:

$$p(\textbf{x}|\textbf{y}) = \frac{p(\textbf{y} | \textbf{x})p(\textbf{x}) }{p(\textbf{y})}$$

Where:
- $p(\textbf{x}|\textbf{y})$ is the *posterior* which allows us to compute the probability of $\textbf{x}$ given the observation of $\textbf{y}$.
- $p(\textbf{y}|\textbf{x})$ is the *likelihood* which tells us how likely our observed $\textbf{y}$ will happen given $\textbf{x}$.
- $p(\textbf{x})$ is the *prior*
- $p(\textbf{y})$ is the *evidence*

We should also be aware of the **Product Rule**  where we can compute the joint probability of $\textbf{x}$ and $\textbf{y}$:

$$
\begin{align*}
p(\textbf{x}, \textbf{y}) &= p(\textbf{x} | \textbf{y})p(\textbf{y}) \\
p(\textbf{x}, \textbf{y}) &= p(\textbf{y} | \textbf{x})p(\textbf{x})
\end{align*}
$$

### Wednesday 7/22

Today we need to implement the Gaussian class:

The PDF formula is:

$$f(x) = \frac{1}{\sqrt{2\pi\sigma^{2}}} e^{-\frac{(x - \mu)^{2}}{2\sigma^{2}}}$$

The Log PDF that returns the log of the density is:

$$
\begin{align*}
\log{f(x)} &= \log \left( \frac{1}{\sqrt{2\pi\sigma^{2}}} e^{-\frac{(x - \mu)^{2}}{2\sigma^{2}}}\right) \\
&= \log \left( (2\pi\sigma^{2})^{-\frac{1}{2}}\right) + \log \left( \exp(- \frac{(x - \mu)^{2}}{2\sigma^{2}})\right) \\
&= -\frac{1}{2}\log(2\pi\sigma^{2}) - \frac{(x - \mu)^{2}}{2\sigma^{2}} \\
&= -\frac{1}{2}[\log(2\pi) + \log(\sigma^{2})] - \frac{(x - \mu)^{2}}{2\sigma^{2}} \\
&= -\frac{1}{2}\log(2\pi) - \frac{1}{2}\log(\sigma^{2}) - \frac{(x - \mu)^{2}}{2\sigma^{2}}
\end{align*}
$$


**Question:** Why are Gaussians so prevalent in ML?

**Answer:** I believe this is due to the Central Limit Theorem, where all random collections of data eventually approach the normal distribution as the number of samples collected (or N) reaches infinity. And since ML's main objective is to see patterns in data distributions, it can often begin with the assumption of a gaussian distribution as the underlying distribution of the dataset given to the model. Of course this doesn't apply in all scenarios and is really just a general assumption.

### Thursday 7/23

What is a mixture exactly?

To make one data point: first flip a weighted coin to pick a cluster (component 1 with probability $\pi_{1}$, component 2 with probability $\pi_{2}$). Then draw a number from that cluster's Gaussian. So now every data point has a secret: which coin-flip made it. 

We'll call that secret $z_i \in {1, 2}$ the *latent variable*. So you always see $x_{i}$ but never $z_{i}$.

$$
p(x) = \underbrace{\pi_1}_{\text{coin}}, \qquad \underbrace{\mathcal{N}(x \mid \mu_1, \sigma_1^2)}_{\text{cluster 1}} + \pi_2, \qquad \mathcal{N}(x \mid \mu_2, \sigma_2^2), \qquad \pi_1 + \pi_2 = 1
$$

So when we look at this geometrically, we see two bell curves, each scaled down by its coin-weight, added toegther. End up with a lumpy curve.

> NOTE: In our day 3 code, on line 151 we used the numerical constant $\pi$ (3.1415...) as opposed to a probability value. When mixing two gaussians, $\pi_1$ and $\pi_2$ represent to probability values that sum to 1. For example $\pi_1 = 0.9$ and $\pi_2 = 0.1$. Need to keep this in mind and fix the bug in yesterday's code.


SO why can't we just use the MLE directly here?

On monday, MLE on a single gaussian was clean because the log killed the exponents:

$$
\log \prod_i \mathcal{N}(x_i\mid\mu,\sigma) = \sum_i \log \mathcal{N}(\dots) = \sum_i \Big[ -\tfrac{(x_i-\mu)^2}{2\sigma^2} - \log\sigma - \dots\Big]
$$

This gives us a clean sum where we can take the derivative and set to zero and get a closed form expression. That the `fit` function pretty much.

Now if we write the log-likelihood for the mixture of gaussians, we can see that break:

$$
\begin{align*}
\log \prod_i \Big[\pi_1\mathcal{N}(x_i\mid\mu_1,\sigma_1) + \pi_2\mathcal{N}(x_i\mid\mu_2,\sigma_2)\Big] = \sum_i \log\Big(\underbrace{\pi_1\mathcal{N}_1 + \pi_2\mathcal{N}2}_{\textbf{sum trapped inside the log}}\Big)
\end{align*}
$$

The log now sits on top of a sum, and $\log(a + b)$ does not simplify and the exp never simplifies. Take $\partial/\partial\mu_1$ and you don't get a closed form; you get an equation where $\mu_1$ depends on how much each point belongs to cluster 1, which also depends on $\mu_1$. This becomes circular and doesn't have an algebraic solution.

**In summary:** In a single gaussian the log distributes over a product and the exp cancels, giving closed-form estimates; in a mixture the sum sits inside the log, the exp survives, and the parameter equations become circular -> each cluster's fit depends on assignments that depend on the fit.

This is pretty much what the **Expectation-Maximization (EM) Algorithm** is trying to resolve, the circularity we described before. If we knew the secret $z_{i}$ (hard labels, "point x came from cluster 1"), the problem becomes like what we did on monday. We can just sort each data point into two piles and run MLE on each separately.

So we need to find the missing $z$:

- Knowing the *parameters* -> we can guess the labels (a point near cluster 1's mean probably came from cluster 1).
- Knowing the *labels* -> You can compute the parameters (run fit on each pile).

So we guess one to compute the other and repeat as it improves each round. Let's break the EM algo down with this insight:

**EM Algorithm:**

*E-step* => "Given my current guess, what cluster owns each data point?" (bayes)

Instead of a hard label, EM computes the probability that point $i$ was made by component $k$, given where it landed.

$$
r_{ik} = P(z_i = k \mid x_i) = \frac{\pi_k,\mathcal{N}(x_i\mid\mu_k,\sigma_k)}{\sum_{j}\pi_j,\mathcal{N}(x_i\mid\mu_j,\sigma_j)}
$$

Lets compare it to our tueday bayesian assignment:

| Bayes (Tues) | EM E-Step |
|-- | -- |
| prior $P(D)$ = Base Rate | prior $\pi_k$ = mixing weight |
| likelihood $P(+ \mid D)$ = sensitivity | likelihood $\mathcal{N}(x_i \mid \mu_k, \sigma_k)$ |
| evidence $P(+)$ = normalizer | denominator $\sum_j\pi_j\mathcal{N}_j$ |
| posterior $P(D\mid +)$ | responsibility $r_{ik}$ |

The term $r_{ik}$ is a fractional term. A point sitting between two clusters might be 0.7 owned by cluster 1 and 0.3 owned by cluster 2. This "soft" ownership is what makes EM differentiable and stable as opposed to hard assignments (like k-means) is jumpy. Each row of responsibilities sums to 1: $r_{i1} + r_{i2} = 1$.

*M-step* => "Given who owns what, re-fit each cluster" (weighted MLE)

Now we can just rerun MLE (like we did monday), but every point contributes to cluster $k$ *in proportion to its responsibility $r_{ik}$*:

$$
\mu_k = \frac{\sum_i r_{ik},x_i}{\sum_i r_{ik}} \qquad \sigma_k^2 = \frac{\sum_i r_{ik},(x_i - \mu_k)^2}{\sum_i r_{ik}} \qquad \pi_k = \frac{\sum_i r_{ik}}{N}
$$

- $\mu_{k}$ = **responsibility-weighted mean**. Points that mostly belong to $k$ have greater influence; barely-owned points don't influence much.
- $\sigma_k$ = **responsibility-weighted variance**. Note the denominator is $\sum_i r_{ik}$, the "effective number of points" cluster $k$ owns.
- $\pi_k$ = **total responsibility mass $\div$ N**. If cluster 1 softly owns 300 of 1000 points, $\pi_1 = 0.3$

This is what the loop looks like:

initialize $\mu$, $\sigma$, $\pi$ ( a guess)  
repeat:  
&emsp;E-step: compute $r_{ik}$ for all i, k  
&emsp;M-step: recompute $\mu_k$, $\sigma_k$, $\pi_k$ from $r$  
until log-likelihood stops increasing

> NOTE: Each E-then-M round **provably does not decrease** the log-likelihood, it monotonically climbs to a fixed point. The proof is **Jensen's inequality** building a lower bound that touches the true likelihood at the current params, then the M-step maximizes that bound. What it DOES NOT guarantee is the global maximum. The mixture of gaussians likelihood is non-convex, so EM lands in whatever basin you started in. A different init -> different answer.



