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



