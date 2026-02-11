# test_corner.py
import numpy as np
import corner
import matplotlib.pyplot as plt

# -----------------------------
# Generate fake 2D samples
# -----------------------------
rng = np.random.default_rng(42)

mean = [-1.0, 0.0]
cov = [[1.0, 0.5],
       [0.5, 1.0]]

samples = rng.multivariate_normal(mean, cov, size=5000)

# -----------------------------
# Truth values
# -----------------------------
truths = [-1.0, 0.0]
fakes = [-0.5, 0.5]  # This will be the line we overplot

# -----------------------------
# Make the corner plot
# -----------------------------
fig = corner.corner(
    samples,
    labels=[r"$\theta_1$", r"$\theta_2$"],
    levels=[0.68, 0.95],
    plot_contours=True,
    fill_contours=True,
    show_titles=True,
    color="C1",
    truths=truths,
    truth_color="k",
)

# -----------------------------
# Overplot lines manually
# -----------------------------
# Shape must be (n_lines, n_dim)
fake_lines = np.array([[0.5], [0.5]])  # This is the line we want to overplot
fake_points = np.array(fake_lines).reshape(1, -1)  # Reshape to (1, 2) for overplot_points
corner.overplot_lines(fig, fake_lines, color="blue", lw=2)
corner.overplot_points(fig, fake_points, color="blue",
marker = "s",  markerfacecolor="blue",markersize = 4, markeredgewidth = 2)

# -----------------------------
# Save / show
# -----------------------------
fig.savefig("test_corner_overplot_lines_with_truths.png", dpi=300)
print("done")
