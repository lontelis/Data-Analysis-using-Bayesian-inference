import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import corner

# ---------------------------
# 0. User settings
# ---------------------------
# We will run for both degree = 1 and degree = 2
degrees = [1, 2]

# Data generation (common to both models)
np.random.seed(42)
xx = np.random.randn(100) * 1.0 + 1.0
yy = xx * 1.0 + np.random.randn(100) * 0.3
yyerr = yy * 0.0 + 0.3          # constant error
cov = np.diag(yyerr**2)
inv_cov = np.linalg.inv(cov)

# Polynomial model factory
def polynomial_model(x, params):
    return sum(params[i] * x**i for i in range(len(params)))

# Chi-squared function
def chi2(params_vector):
    return np.array([((yy - polynomial_model(xx, p)) @ inv_cov @ (yy - polynomial_model(xx, p))) for p in params_vector])

# Function to run the full analysis for a given degree
def run_analysis(degree):
    print(f"\n{'='*50}")
    print(f"Running analysis for degree = {degree} (polynomial order {degree})")
    print('='*50)
    
    n_params = degree + 1
    param_names = [f'$p_{i}$' for i in range(n_params)]
    base_fname = f"_deg{degree}"

    # Prior means and sigmas
    prior_means = [0.0, 1.0] + [0.0] * (degree - 1)
    prior_sigmas = [1.5] * n_params

    # ----- Prior only -----
    with pm.Model() as prior_model:
        params = []
        for i in range(n_params):
            param = pm.Normal(param_names[i], mu=prior_means[i], sigma=prior_sigmas[i])
            params.append(param)
        prior_samples = pm.sample_prior_predictive(samples=5000, return_inferencedata=True)

    # ----- Data only (flat priors) -----
    with pm.Model() as data_model:
        params = []
        for i in range(n_params):
            param = pm.Uniform(param_names[i], lower=-10, upper=10)
            params.append(param)
        mu = polynomial_model(xx, params)
        likelihood = pm.Normal('y', mu=mu, sigma=yyerr, observed=yy)
        trace_data = pm.sample(draws=2000, tune=1000, chains=4, return_inferencedata=True,
                               progressbar=False)

    # ----- Data + Prior -----
    with pm.Model() as full_model:
        params = []
        for i in range(n_params):
            param = pm.Normal(param_names[i], mu=prior_means[i], sigma=prior_sigmas[i])
            params.append(param)
        mu = polynomial_model(xx, params)
        likelihood = pm.Normal('y', mu=mu, sigma=yyerr, observed=yy)
        trace_full = pm.sample(draws=2000, tune=1000, chains=4, return_inferencedata=True,
                               progressbar=False)

    # Convergence diagnostics
    def check_convergence(trace, name):
        rhat = az.rhat(trace)
        print(f"\n{name} convergence (R-hat):")
        all_ok = True
        rhat_dict = {}
        for p in param_names:
            val = rhat[p].values
            ok = val < 1.01
            rhat_dict[p] = (val, ok)
            print(f"  {p} R-hat = {val:.4f}  {'✅' if ok else '❌'}")
            all_ok = all_ok and ok
        return rhat_dict

    rhat_data = check_convergence(trace_data, "Data-only model")
    rhat_full = check_convergence(trace_full, "Full model (Data+Prior)")

    # Trace plots with R-hat annotation
    def plot_trace_with_rhat(trace, rhat_dict, title, filename):
        axes = az.plot_trace(trace, var_names=param_names, compact=True,
                             legend=True, figsize=(12, 2*n_params))
        plt.suptitle(title)
        # Add R-hat text
        fig = plt.gcf()
        text_str = "R-hat: " + ", ".join([f"{p}={v[0]:.3f}{'✅' if v[1] else '❌'}" for p, v in rhat_dict.items()])
        fig.text(0.1, 0.95, text_str, fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        plt.tight_layout()
        plt.savefig(filename)
        plt.show()

    plot_trace_with_rhat(trace_data, rhat_data, f'Trace plots – Data-only model (deg {degree})', f'trace_data{base_fname}.pdf')
    plot_trace_with_rhat(trace_full, rhat_full, f'Trace plots – Full model (Data+Prior) (deg {degree})', f'trace_full{base_fname}.pdf')

    # Extract chains
    def extract_chains(trace_or_prior, source='posterior'):
        chains = []
        for p in param_names:
            if source == 'posterior':
                arr = trace_or_prior.posterior[p].values.flatten()
            else:
                arr = trace_or_prior.prior[p].values.flatten()
            chains.append(arr)
        return np.column_stack(chains)

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

    # Plot data with best-fit lines
    x_plot = np.linspace(xx.min()-0.5, xx.max()+0.5, 100)
    plt.figure(figsize=(10, 6))
    plt.errorbar(xx, yy, yerr=yyerr, fmt='o', capsize=2, label='Data', alpha=0.7, color='gray')
    plt.plot(x_plot, polynomial_model(x_plot, mean_prior), 'g-', linewidth=2, label='Prior')
    plt.plot(x_plot, polynomial_model(x_plot, mean_data), 'b-', linewidth=2, label='Data')
    plt.plot(x_plot, polynomial_model(x_plot, mean_full), 'r-', linewidth=2, label='Data+Prior')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Data and best‑fit models (deg {degree})')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(f'best_fit_lines{base_fname}.pdf')
    plt.show()

    # Compute chi² for all samples
    chi2_prior = chi2(samples_prior)
    chi2_data = chi2(samples_data)
    chi2_full = chi2(samples_full)

    # Summarize each case
    def summarize_samples(samples, chi2_vals, case_name):
        idx_min = np.argmin(chi2_vals)
        best_params = samples[idx_min]
        best_chi2 = chi2_vals[idx_min]
        mean_params = np.mean(samples, axis=0)
        std_params = np.std(samples, axis=0, ddof=1)
        intervals = {}
        for i, p in enumerate(param_names):
            p16, p50, p84 = np.percentile(samples[:, i], [16, 50, 84])
            p2_5, p97_5 = np.percentile(samples[:, i], [2.5, 97.5])
            intervals[p] = {'p16': p16, 'p50': p50, 'p84': p84,
                            'p2_5': p2_5, 'p97_5': p97_5}
        print(f"\n{case_name}:")
        print(f"  Best-fit (min χ²): {', '.join([f'{p}={v:.3f}' for p, v in zip(param_names, best_params)])}, χ²_min = {best_chi2:.2f}")
        print(f"  Mean ± 1σ: {', '.join([f'{p}={mean_params[i]:.3f} ± {std_params[i]:.3f}' for i, p in enumerate(param_names)])}")
        return best_params, best_chi2, mean_params, std_params, intervals

    prior_summary = summarize_samples(samples_prior, chi2_prior, "Prior")
    data_summary = summarize_samples(samples_data, chi2_data, "Data")
    full_summary = summarize_samples(samples_full, chi2_full, "Data+Prior")

    # Corner plot with legend showing mean ± 1σ
    def format_label(name, mean_params, std_params):
        param_str = ", ".join([f"{p}={mean_params[i]:.2f}±{std_params[i]:.2f}" for i, p in enumerate(param_names)])
        return f"{name}: {param_str}"

    labels = [
        format_label("Prior", prior_summary[2], prior_summary[3]),
        format_label("Data", data_summary[2], data_summary[3]),
        format_label("Data+Prior", full_summary[2], full_summary[3])
    ]

    samples_list = [samples_prior, samples_data, samples_full]
    colors = ['green', 'blue', 'red']

    fig = corner.corner(samples_list[0], labels=param_names, color=colors[0],
                        hist_kwargs={'density':True}, label=labels[0],
                        show_titles=True, title_fmt='.3f',
                        levels=(1 - np.exp(-0.5), 1 - np.exp(-2)))
    corner.corner(samples_list[1], fig=fig, color=colors[1], hist_kwargs={'density':True},
                  label=labels[1], levels=(1 - np.exp(-0.5), 1 - np.exp(-2)))
    corner.corner(samples_list[2], fig=fig, color=colors[2], hist_kwargs={'density':True},
                  label=labels[2], levels=(1 - np.exp(-0.5), 1 - np.exp(-2)))

    handles = [plt.Line2D([0], [0], color=c, lw=2) for c in colors]
    plt.legend(handles, labels, loc='upper right', frameon=True, fontsize=8,
               bbox_to_anchor=(1.0, 2.0))
    plt.savefig(f'corner_plot{base_fname}.pdf', bbox_inches='tight')
    plt.show()

    # Chi² scatter plots
    def plot_chi2_scatter(samples, chi2_vals, case_name, color, summary):
        best_params, best_chi2, mean_params, std_params, intervals = summary
        n_p = samples.shape[1]
        fig, axes = plt.subplots(n_p, 1, figsize=(8, 3*n_p), sharex=False)
        if n_p == 1:
            axes = [axes]
        chi2_min = chi2_vals.min()
        for i, ax in enumerate(axes):
            pname = param_names[i]
            ax.scatter(samples[:, i], chi2_vals, s=5, alpha=0.3, c=color)
            ax.scatter(best_params[i], best_chi2, s=80, c='black', marker='*',
                       edgecolors='white', zorder=5, label=f'Best fit: {pname}={best_params[i]:.3f}')
            ax.set_xlabel(pname)
            ax.set_ylabel('χ²')
            ax.set_title(f'χ² vs {pname} – {case_name} (deg {degree})')
            ax.axhline(y=chi2_min, color='k', linestyle='-', linewidth=1, label='χ²$_{min}$')
            ax.axhline(y=chi2_min + 1, color='k', linestyle='--', linewidth=1, label='χ²$_{min}$ + 1 (1σ)')
            ax.axhline(y=chi2_min + 4, color='k', linestyle=':', linewidth=1, label='χ²$_{min}$ + 4 (2σ)')
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

    # Return the best chi2 (from data-only case) and number of parameters for model comparison
    return {
        'degree': degree,
        'best_chi2': data_summary[1],   # chi2_min for data-only (likelihood)
        'n_params': n_params,
        'n_data': len(xx)
    }

# ---------------------------
# Run for both degrees and collect results
# ---------------------------
results = {}
for d in degrees:
    results[d] = run_analysis(d)

# ---------------------------
# Model Comparison
# ---------------------------
print("\n" + "="*50)
print("MODEL COMPARISON: Linear (degree=1) vs Quadratic (degree=2)")
print("="*50)

n = results[1]['n_data']  # same for both
comparison = []
for d in degrees:
    k = results[d]['n_params']
    chi2_min = results[d]['best_chi2']
    # For Gaussian errors, -2 ln L = chi2_min + constant (constant = sum ln(2πσ_i²))
    # This constant is the same for all models, so it cancels in model comparison.
    # Therefore we can use chi2_min + 2k as AIC, etc.
    aic = 2*k + chi2_min
    aicc = aic + (2*k*(k+1))/(n - k - 1)
    bic = k*np.log(n) + chi2_min
    reduced_chi2 = chi2_min / (n - k)
    comparison.append({
        'degree': d,
        'k': k,
        'chi2_min': chi2_min,
        'reduced_chi2': reduced_chi2,
        'AIC': aic,
        'AICc': aicc,
        'BIC': bic
    })

# Print LaTeX table
print("\nLaTeX table for inclusion in document:\n")
print("\\begin{table}[h!]")
print("\\centering")
print("\\caption{Model comparison for linear ($k=2$) vs. quadratic ($k=3$) fit.}")
print("\\label{tab:modelcomp}")
print("\\begin{tabular}{lcccccc}")
print("\\toprule")
print("Model & $k$ & $\\chi^2_{\\min}$ & $\\chi^2/\\nu$ & AIC & AICc & BIC \\\\")
print("\\midrule")
for res in comparison:
    model_name = f"deg {res['degree']}"
    print(f"{model_name:8} & {res['k']} & {res['chi2_min']:.2f} & {res['reduced_chi2']:.3f} & {res['AIC']:.2f} & {res['AICc']:.2f} & {res['BIC']:.2f} \\\\")
print("\\bottomrule")
print("\\end{tabular}")
print("\\end{table}")

# Interpretation
print("\nInterpretation:")
print("  - Lower AIC/BIC indicates a better model.")
print("  - ΔAIC/BIC > 2 suggests substantial support for the model with the lower value.")
print("  - χ²/ν near 1 indicates a good fit.")
print("\nBased on the values above, the preferred model is the one with the smallest AICc and BIC.")
print("Typically, we select the model with the lowest information criterion.")

# Also print a plain text table for console
print("\nPlain text summary:")
print(f"\nNumber of data points n = {n}")
print("\n{:<10} {:<5} {:<10} {:<15} {:<10} {:<12} {:<10}".format(
    "Model", "k", "χ²_min", "χ²/ν", "AIC", "AICc", "BIC"))
print("-"*70)
for res in comparison:
    print("{:<10} {:<5} {:<10.2f} {:<15.3f} {:<10.2f} {:<12.2f} {:<10.2f}".format(
        f"deg {res['degree']}", res['k'], res['chi2_min'], res['reduced_chi2'],
        res['AIC'], res['AICc'], res['BIC']))