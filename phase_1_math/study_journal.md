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
