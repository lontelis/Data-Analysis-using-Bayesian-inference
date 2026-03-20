import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import corner

# ---------------------------
# 0. User settings
# ---------------------------
degree = 2                     # polynomial degree (1 = linear, 2 = quadratic, ...)
n_params = degree + 1
param_names = [f'$p_{i}$' for i in range(n_params)]   # LaTeX formatting

# Prior means and sigmas for each parameter
#prior_means = [0.0, 1.0] + [0.0] * (degree - 1)   # p0=0, p1=1, higher=0
#prior_sigmas = [1.5] * n_params                     # all have sigma=1.5

# Prior means and sigmas for each parameter play with prior
prior_means = [0.0, 1.0] + [0.1] * (degree)   # p0=0, p1=1, higher=0
prior_sigmas = [0.05] * n_params                     # all have sigma=1.5

# Base filename for saving (includes degree)
base_fname = f"_deg{degree}"

# ---------------------------
# 1. Generate synthetic data (still linear for demonstration)
# ---------------------------
np.random.seed(42)
xx = np.random.randn(100) * 1.0 + 1.0
yy = xx * 1.0 + np.random.randn(100) * 0.3
yyerr = yy * 0.0 + 0.3          # constant error

# Covariance matrix (diagonal)
cov = np.diag(yyerr**2)
inv_cov = np.linalg.inv(cov)

# ---------------------------
# 2. Polynomial model factory
# ---------------------------
def polynomial_model(x, params):
    """Evaluate polynomial of given degree at x."""
    return sum(params[i] * x**i for i in range(len(params)))

# ---------------------------
# 3. Define the three cases
# ---------------------------
# Prior only (no data)
with pm.Model() as prior_model:
    params = []
    for i in range(n_params):
        param = pm.Normal(param_names[i], mu=prior_means[i], sigma=prior_sigmas[i])
        params.append(param)
    prior_samples = pm.sample_prior_predictive(samples=5000, return_inferencedata=True)

# Data only (flat priors) – using Uniform over wide range
with pm.Model() as data_model:
    params = []
    for i in range(n_params):
        # Use wide uniform prior (effectively flat)
        param = pm.Uniform(param_names[i], lower=-10, upper=10)
        params.append(param)
    # Build linear predictor
    mu = polynomial_model(xx, params)
    likelihood = pm.Normal('y', mu=mu, sigma=yyerr, observed=yy)
    trace_data = pm.sample(draws=2000, tune=1000, chains=4, return_inferencedata=True,
                           progressbar=False)

# Data + Prior (informative priors)
with pm.Model() as full_model:
    params = []
    for i in range(n_params):
        param = pm.Normal(param_names[i], mu=prior_means[i], sigma=prior_sigmas[i])
        params.append(param)
    mu = polynomial_model(xx, params)
    likelihood = pm.Normal('y', mu=mu, sigma=yyerr, observed=yy)
    trace_full = pm.sample(draws=2000, tune=1000, chains=4, return_inferencedata=True,
                           progressbar=False)

# ---------------------------
# 4. Convergence diagnostics
# ---------------------------
def check_convergence(trace, name):
    rhat = az.rhat(trace)
    print(f"\n{name} convergence (R-hat):")
    all_ok = True
    for p in param_names:
        val = rhat[p].values
        ok = val < 1.01
        print(f"  {p} R-hat = {val:.4f}  {'✅' if ok else '❌'}")
        all_ok = all_ok and ok
    return all_ok

print("="*50)
check_convergence(trace_data, "Data-only model")
check_convergence(trace_full, "Full model (Data+Prior)")
print("="*50)

# Trace plots for all parameters
az.plot_trace(trace_data, var_names=param_names, compact=True,
              legend=True, figsize=(12, 2*n_params))
plt.suptitle('Trace plots – Data-only model')
plt.tight_layout()
plt.savefig(f'trace_data{base_fname}.pdf')
plt.show()

az.plot_trace(trace_full, var_names=param_names, compact=True,
              legend=True, figsize=(12, 2*n_params))
plt.suptitle('Trace plots – Full model (Data+Prior)')
plt.tight_layout()
plt.savefig(f'trace_full{base_fname}.pdf')
plt.show()

# ---------------------------
# 5. Extract chains for all parameters
# ---------------------------
def extract_chains(trace_or_prior, source='posterior'):
    """Extract flattened chains for all parameters."""
    chains = []
    for p in param_names:
        if source == 'posterior':
            arr = trace_or_prior.posterior[p].values.flatten()
        else:  # prior predictive
            arr = trace_or_prior.prior[p].values.flatten()
        chains.append(arr)
    return np.column_stack(chains)   # shape (n_samples, n_params)

samples_prior = extract_chains(prior_samples, source='prior')
samples_data = extract_chains(trace_data, source='posterior')
samples_full = extract_chains(trace_full, source='posterior')

# Means for best-fit lines
mean_prior = np.mean(samples_prior, axis=0)
mean_data = np.mean(samples_data, axis=0)
mean_full = np.mean(samples_full, axis=0)

print("\nMean values:")
print(f"  Prior:        {', '.join([f'{p}={v:.3f}' for p, v in zip(param_names, mean_prior)])}")
print(f"  Data:         {', '.join([f'{p}={v:.3f}' for p, v in zip(param_names, mean_data)])}")
print(f"  Data+Prior:   {', '.join([f'{p}={v:.3f}' for p, v in zip(param_names, mean_full)])}")

# ---------------------------
# 6. Plot data with error bars and best-fit lines
# ---------------------------
x_plot = np.linspace(xx.min()-0.5, xx.max()+0.5, 100)

plt.figure(figsize=(10, 6))
plt.errorbar(xx, yy, yerr=yyerr, fmt='o', capsize=2, label='Data', alpha=0.7, color='gray')
plt.plot(x_plot, polynomial_model(x_plot, mean_prior), 'g-', linewidth=2, label='Prior')
plt.plot(x_plot, polynomial_model(x_plot, mean_data), 'b-', linewidth=2, label='Data')
plt.plot(x_plot, polynomial_model(x_plot, mean_full), 'r-', linewidth=2, label='Data+Prior')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Data and best‑fit models')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(f'best_fit_lines{base_fname}.pdf')
plt.show()

# ---------------------------
# 7. Compute χ² for all samples
# ---------------------------
def chi2(params_vector):
    return np.array([((yy - polynomial_model(xx, p)) @ inv_cov @ (yy - polynomial_model(xx, p))) for p in params_vector])

chi2_prior = chi2(samples_prior)
chi2_data = chi2(samples_data)
chi2_full = chi2(samples_full)

# ---------------------------
# 8. Summarize each case: best-fit, mean, std, intervals
# ---------------------------
def summarize_samples(samples, chi2_vals, case_name):
    n_samp, n_p = samples.shape
    idx_min = np.argmin(chi2_vals)
    best_params = samples[idx_min]
    best_chi2 = chi2_vals[idx_min]
    
    # Mean and standard deviation
    mean_params = np.mean(samples, axis=0)
    std_params = np.std(samples, axis=0, ddof=1)   # sample standard deviation
    
    # Percentiles for each parameter
    intervals = {}
    for i, p in enumerate(param_names):
        p16, p50, p84 = np.percentile(samples[:, i], [16, 50, 84])
        p2_5, p97_5 = np.percentile(samples[:, i], [2.5, 97.5])
        intervals[p] = {
            'p16': p16, 'p50': p50, 'p84': p84,
            'p2_5': p2_5, 'p97_5': p97_5
        }
    
    print(f"\n{case_name}:")
    print(f"  Best-fit (min χ²): {', '.join([f'{p}={v:.3f}' for p, v in zip(param_names, best_params)])}, χ²_min = {best_chi2:.2f}")
    print(f"  Mean ± 1σ: {', '.join([f'{p}={mean_params[i]:.3f} ± {std_params[i]:.3f}' for i, p in enumerate(param_names)])}")
    for p in param_names:
        print(f"  {p} 68% interval : [{intervals[p]['p16']:.3f}, {intervals[p]['p84']:.3f}] (median={intervals[p]['p50']:.3f})")
        print(f"  {p} 95% interval : [{intervals[p]['p2_5']:.3f}, {intervals[p]['p97_5']:.3f}]")
    
    return best_params, best_chi2, mean_params, std_params, intervals

prior_summary = summarize_samples(samples_prior, chi2_prior, "Prior")
data_summary = summarize_samples(samples_data, chi2_data, "Data")
full_summary = summarize_samples(samples_full, chi2_full, "Data+Prior")

# ---------------------------
# 9. Corner plot with legend showing mean ± 1σ
# ---------------------------
def format_label(name, mean_params, std_params):
    # Create a single line with all parameters: e.g., "Prior: p0=0.02±0.15, p1=1.03±0.08"
    param_str = ", ".join([f"{p}={mean_params[i]:.2f}±{std_params[i]:.2f}" for i, p in enumerate(param_names)])
    return f"{name}: {param_str}"

labels = [
    format_label("Prior", prior_summary[2], prior_summary[3]),
    format_label("Data", data_summary[2], data_summary[3]),
    format_label("Data+Prior", full_summary[2], full_summary[3])
]

# For corner, we need to pass the samples as list of arrays (each shape (n, n_params))
samples_list = [samples_prior, samples_data, samples_full]
colors = ['green', 'blue', 'red']

fig = corner.corner(samples_list[0], labels=param_names, color=colors[0],
                    hist_kwargs={'density':True}, label=labels[0],
                    show_titles=True, title_fmt='.3f',   # titles show median and 16th/84th percentiles
                    levels=(1 - np.exp(-0.5), 1 - np.exp(-2)))   # 1σ and 2σ contours
corner.corner(samples_list[1], fig=fig, color=colors[1], hist_kwargs={'density':True},
              label=labels[1], levels=(1 - np.exp(-0.5), 1 - np.exp(-2)))
corner.corner(samples_list[2], fig=fig, color=colors[2], hist_kwargs={'density':True},
              label=labels[2], levels=(1 - np.exp(-0.5), 1 - np.exp(-2)))

# Create custom legend handles and place it in the upper-right empty region
handles = [plt.Line2D([0], [0], color=c, lw=2) for c in colors]
# Updated legend placement as requested
plt.legend(handles, labels, loc='upper right', frameon=True, fontsize=8,
           bbox_to_anchor=(1.0, 2.0))
plt.savefig(f'corner_plot{base_fname}.pdf', bbox_inches='tight')
plt.show()

# ---------------------------
# 10. Scatter plots of χ² vs each parameter
# ---------------------------
def plot_chi2_scatter(samples, chi2_vals, case_name, color, summary):
    # Unpack summary: best_params, best_chi2, mean_params, std_params, intervals
    best_params, best_chi2, mean_params, std_params, intervals = summary
    n_p = samples.shape[1]
    # Create subplots: one column per parameter
    fig, axes = plt.subplots(n_p, 1, figsize=(8, 3*n_p), sharex=False)
    if n_p == 1:
        axes = [axes]
    
    chi2_min = chi2_vals.min()
    for i, ax in enumerate(axes):
        pname = param_names[i]
        ax.scatter(samples[:, i], chi2_vals, s=5, alpha=0.3, c=color)
        # Mark best-fit point
        ax.scatter(best_params[i], best_chi2, s=80, c='black', marker='*',
                   edgecolors='white', zorder=5, label=f'Best fit: {pname}={best_params[i]:.3f}')
        ax.set_xlabel(pname)
        ax.set_ylabel('χ²')
        ax.set_title(f'χ² vs {pname} – {case_name}')
        # Use LaTeX formatting for thresholds
        ax.axhline(y=chi2_min, color='k', linestyle='-', linewidth=1, label='χ²$_{min}$')
        ax.axhline(y=chi2_min + 1, color='k', linestyle='--', linewidth=1, label='χ²$_{min}$ + 1 (1σ)')
        ax.axhline(y=chi2_min + 4, color='k', linestyle=':', linewidth=1, label='χ²$_{min}$ + 4 (2σ)')
        # Add text box with intervals (using percentile intervals)
        low68 = intervals[pname]['p16']
        high68 = intervals[pname]['p84']
        low95 = intervals[pname]['p2_5']
        high95 = intervals[pname]['p97_5']
        textstr = f'68%: [{low68:.3f}, {high68:.3f}]\n95%: [{low95:.3f}, {high95:.3f}]'
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'chi2_scatter_{case_name.lower().replace("+","")}{base_fname}.pdf')
    plt.show()

plot_chi2_scatter(samples_prior, chi2_prior, 'Prior', 'green', prior_summary)
plot_chi2_scatter(samples_data, chi2_data, 'Data', 'blue', data_summary)
plot_chi2_scatter(samples_full, chi2_full, 'Data+Prior', 'red', full_summary)

print(f"\nAll plots saved as PDF files with suffix '{base_fname}'.")