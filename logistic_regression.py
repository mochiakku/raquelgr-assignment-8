import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from scipy.spatial.distance import cdist
import os

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

def generate_ellipsoid_clusters(distance, n_samples=100, cluster_std=0.5):
    np.random.seed(0)
    covariance_matrix = np.array([[cluster_std, cluster_std * 0.8], 
                                 [cluster_std * 0.8, cluster_std]])
    
    # Generate the first cluster (class 0)
    X1 = np.random.multivariate_normal(mean=[1, 1], cov=covariance_matrix, size=n_samples)
    y1 = np.zeros(n_samples)

    # Generate the second cluster (class 1)
    X2 = np.random.multivariate_normal(mean=[1, 1], cov=covariance_matrix, size=n_samples)
    # Shift the second cluster diagonally by the specified distance
    X2[:, 0] += distance  # Shift along x-axis
    X2[:, 1] -= distance  # Shift along y-axis
    y2 = np.ones(n_samples)

    # Combine the clusters into one dataset
    X = np.vstack((X1, X2))
    y = np.hstack((y1, y2))
    return X, y

def calculate_logistic_loss(model, X, y):
    # Calculate probabilities
    probs = model.predict_proba(X)
    # Calculate log loss (negative log likelihood)
    epsilon = 1e-15  # Small constant to avoid log(0)
    probs = np.clip(probs, epsilon, 1 - epsilon)
    loss = -np.mean(y * np.log(probs[:, 1]) + (1 - y) * np.log(probs[:, 0]))
    return loss

def do_experiments(start, end, step_num):
    plt.style.use('default')  # Use default style instead of seaborn
    
    # Set up experiment parameters
    shift_distances = np.linspace(start, end, step_num)
    beta0_list, beta1_list, beta2_list = [], [], []
    slope_list, intercept_list, loss_list, margin_widths = [], [], [], []

    n_samples = step_num
    n_cols = 2
    n_rows = (n_samples + n_cols - 1) // n_cols
    
    # Create figure for datasets
    fig_datasets = plt.figure(figsize=(20, n_rows * 10))
    fig_datasets.patch.set_facecolor('white')  # Set white background

    # Run experiments for each shift distance
    for i, distance in enumerate(shift_distances, 1):
        # Generate dataset and fit model
        X, y = generate_ellipsoid_clusters(distance=distance)
        model = LogisticRegression()
        model.fit(X, y)
        
        # Extract and store parameters
        beta0 = model.intercept_[0]
        beta1, beta2 = model.coef_[0]
        beta0_list.append(beta0)
        beta1_list.append(beta1)
        beta2_list.append(beta2)
        
        # Calculate and store slope and intercept
        slope = -beta1 / beta2
        intercept = -beta0 / beta2
        slope_list.append(slope)
        intercept_list.append(intercept)
        
        # Calculate and store logistic loss
        loss = calculate_logistic_loss(model, X, y)
        loss_list.append(loss)

        # Plot dataset
        ax = plt.subplot(n_rows, n_cols, i)
        ax.set_facecolor('white')  # Set white background for subplot
        plt.grid(True, linestyle='--', alpha=0.7)  # Add grid
        plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='blue', label='Class 0', alpha=0.6)
        plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='red', label='Class 1', alpha=0.6)
        
        # Plot decision boundary and confidence contours
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
        Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        Z = Z.reshape(xx.shape)

        # Plot confidence contours
        contour_levels = [0.7, 0.8, 0.9]
        alphas = [0.05, 0.1, 0.15]
        for level, alpha in zip(contour_levels, alphas):
            class_1_contour = plt.contourf(xx, yy, Z, levels=[level, 1.0], colors=['red'], alpha=alpha)
            class_0_contour = plt.contourf(xx, yy, Z, levels=[0.0, 1 - level], colors=['blue'], alpha=alpha)
            if level == 0.7:
                distances = cdist(class_1_contour.collections[0].get_paths()[0].vertices,
                                class_0_contour.collections[0].get_paths()[0].vertices,
                                metric='euclidean')
                min_distance = np.min(distances)
                margin_widths.append(min_distance)

        # Plot decision boundary
        plt.contour(xx, yy, Z, levels=[0.5], colors='k', linestyles='--')
        
        plt.title(f"Shift Distance = {distance:.2f}", fontsize=24)
        plt.xlabel("x1", fontsize=14)
        plt.ylabel("x2", fontsize=14)

        # Display decision boundary equation and margin width
        equation_text = f"{beta0:.2f} + {beta1:.2f} * x1 + {beta2:.2f} * x2 = 0\nx2 = {slope:.2f} * x1 + {intercept:.2f}"
        margin_text = f"Margin Width: {min_distance:.2f}"
        plt.text(x_min + 0.1, y_max - 1.0, equation_text, fontsize=24, color="black",
                ha='left', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
        plt.text(x_min + 0.1, y_max - 1.5, margin_text, fontsize=24, color="black",
                ha='left', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

        if i == 1:
            plt.legend(loc='lower right', fontsize=20)

    plt.tight_layout()
    fig_datasets.savefig(f"{result_dir}/dataset.png", dpi=300, bbox_inches='tight')
    plt.close(fig_datasets)

    # Create parameter plots
    fig_params = plt.figure(figsize=(18, 15))
    fig_params.patch.set_facecolor('white')

    # Plot beta0
    plt.subplot(3, 3, 1)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.plot(shift_distances, beta0_list, 'b-', marker='o')
    plt.title("Shift Distance vs Beta0", fontsize=12)
    plt.xlabel("Shift Distance", fontsize=10)
    plt.ylabel("Beta0", fontsize=10)

    # Plot beta1
    plt.subplot(3, 3, 2)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.plot(shift_distances, beta1_list, 'r-', marker='o')
    plt.title("Shift Distance vs Beta1 (Coefficient for x1)", fontsize=12)
    plt.xlabel("Shift Distance", fontsize=10)
    plt.ylabel("Beta1", fontsize=10)

    # Plot beta2
    plt.subplot(3, 3, 3)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.plot(shift_distances, beta2_list, 'g-', marker='o')
    plt.title("Shift Distance vs Beta2 (Coefficient for x2)", fontsize=12)
    plt.xlabel("Shift Distance", fontsize=10)
    plt.ylabel("Beta2", fontsize=10)

    # Plot slope (beta1/beta2)
    plt.subplot(3, 3, 4)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.plot(shift_distances, slope_list, 'm-', marker='o')
    plt.title("Shift Distance vs Beta1/Beta2 (Slope)", fontsize=12)
    plt.xlabel("Shift Distance", fontsize=10)
    plt.ylabel("Beta1/Beta2", fontsize=10)
    plt.ylim(-2, 0)

    # Plot intercept ratio (beta0/beta2)
    plt.subplot(3, 3, 5)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.plot(shift_distances, [-b0/b2 for b0, b2 in zip(beta0_list, beta2_list)], 'c-', marker='o')
    plt.title("Shift Distance vs Beta0/Beta2 (Intercept Ratio)", fontsize=12)
    plt.xlabel("Shift Distance", fontsize=10)
    plt.ylabel("Beta0/Beta2", fontsize=10)

    # Plot logistic loss
    plt.subplot(3, 3, 6)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.plot(shift_distances, loss_list, 'y-', marker='o')
    plt.title("Shift Distance vs Logistic Loss", fontsize=12)
    plt.xlabel("Shift Distance", fontsize=10)
    plt.ylabel("Logistic Loss", fontsize=10)

    # Plot margin width
    plt.subplot(3, 3, 7)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.plot(shift_distances, margin_widths, 'k-', marker='o')
    plt.title("Shift Distance vs Margin Width", fontsize=12)
    plt.xlabel("Shift Distance", fontsize=10)
    plt.ylabel("Margin Width", fontsize=10)

    plt.tight_layout()
    fig_params.savefig(f"{result_dir}/parameters_vs_shift_distance.png", dpi=300, bbox_inches='tight')
    plt.close(fig_params)

if __name__ == "__main__":
    start = 0.25
    end = 2.0
    step_num = 8
    do_experiments(start, end, step_num)