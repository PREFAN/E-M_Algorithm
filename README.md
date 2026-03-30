# E-M_Algorithm
This Python script implements the Expectation-Maximization (EM) Algorithm from scratch to solve a classic Gaussian Mixture Model (GMM) problem. The project uses a small dataset of eight exam scores — [50, 55, 60, 65, 70, 85, 90, 95] — and models them as coming from two underlying Gaussian distributions, representing two student groups: one strong in English and the other strong in Math.
The code is divided into three main sections that progressively build understanding of the EM algorithm.
First, it performs a simple initial clustering by assigning each score to the nearest of two initial mean guesses (55 and 85). It then calculates the variance of each resulting cluster. This serves as a warm-up to show the starting point before applying the full probabilistic approach.
The second section implements the Expectation (E) Step. Using initial parameters (means, variances, and mixing coefficients of 0.5 each), the script calculates the probability density of each data point under both Gaussian distributions. It then computes the responsibilities (gamma values) — the posterior probability that each data point belongs to each cluster. These responsibilities are displayed in a detailed pandas DataFrame, showing likelihoods, numerators, and final soft assignments.
The third and most important section runs the complete EM Algorithm for 5 iterations. In each iteration, the code performs:

E-Step: Calculates responsibilities based on current parameters.
M-Step: Updates the parameters (means, variances, and mixing coefficients) to maximize the expected log-likelihood.

The script tracks and displays how the parameters evolve across iterations, along with the log-likelihood value, which should increase with each step as the model improves. Finally, it presents a clean summary table showing the converged parameters for both clusters: mean, standard deviation, and mixing proportion.
This implementation is highly educational because it avoids using high-level libraries like scikit-learn’s GaussianMixture. Instead, it manually codes the mathematical formulas for Gaussian PDF, responsibility calculation, and parameter updates. It also includes safeguards (small epsilon values) to prevent division by zero or log of zero errors.
Overall, the project provides a clear, step-by-step walkthrough of one of the most important algorithms in unsupervised machine learning. It helps students understand how the EM algorithm iteratively finds maximum likelihood estimates for Gaussian Mixture Models when dealing with latent variables (hidden cluster assignments).
