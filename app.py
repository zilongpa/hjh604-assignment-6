import base64
import io
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, url_for
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend for non-GUI rendering

app = Flask(__name__)


def generate_plots(N, mu, sigma2, S):
    rng = np.random.default_rng()
    
    X = rng.uniform(0, 1, N).reshape(-1, 1)
    Y = rng.normal(mu, np.sqrt(sigma2), N)

    model = LinearRegression()
    model.fit(X, Y)
    slope = model.coef_[0]
    intercept = model.intercept_

    plt.scatter(X, Y, c="blue")
    line_x = np.linspace(0, 1, 100).reshape(-1, 1)
    line_y = model.predict(line_x)

    plt.plot(line_x, line_y, color='red')

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"Linear Fit: y = {intercept:.2f} + {slope:.2f}x")
    
    plot1_path = "static/plot1.png"
    plt.savefig(plot1_path)
    plt.close()


    slopes = []
    intercepts = []

    for _ in range(S):
        X_sim = rng.uniform(0, 1, N).reshape(-1, 1)
        Y_sim =  rng.normal(mu, np.sqrt(sigma2), N)

        sim_model = LinearRegression()
        sim_model.fit(X_sim, Y_sim)

        slopes.append(sim_model.coef_[0])
        intercepts.append(sim_model.intercept_)

    # Plot histograms of slopes and intercepts
    plt.figure(figsize=(10, 5))
    plt.hist(slopes, bins=20, alpha=0.5, color="blue", label="Slopes")
    plt.hist(intercepts, bins=20, alpha=0.5,
             color="orange", label="Intercepts")
    plt.axvline(slope, color="blue", linestyle="--",
                linewidth=1, label=f"Slope: {slope:.2f}")
    plt.axvline(intercept, color="orange", linestyle="--",
                linewidth=1, label=f"Intercept: {intercept:.2f}")
    plt.title("Histogram of Slopes and Intercepts")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()
    plot2_path = "static/plot2.png"
    plt.savefig(plot2_path)
    plt.close()

    # Below code is already provided
    # Calculate proportions of more extreme slopes and intercepts
    # For slopes, we will count how many are greater than the initial slope; for intercepts, count how many are less.
    slope_more_extreme = sum(s > slope for s in slopes) / S  # Already provided
    intercept_more_extreme = sum(
        i < intercept for i in intercepts) / S  # Already provided

    return plot1_path, plot2_path, slope_more_extreme, intercept_more_extreme


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get user input
        N = int(request.form["N"])
        mu = float(request.form["mu"])
        sigma2 = float(request.form["sigma2"])
        S = int(request.form["S"])

        # Generate plots and results
        plot1, plot2, slope_extreme, intercept_extreme = generate_plots(
            N, mu, sigma2, S)

        return render_template("index.html", plot1=plot1, plot2=plot2,
                               slope_extreme=slope_extreme, intercept_extreme=intercept_extreme)

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
