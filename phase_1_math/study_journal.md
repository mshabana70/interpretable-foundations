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

Answer: for my implementation, it is a time complexity of $O(n^2)$