# %% [markdown]
# # Portfolio Optimization Project
# ## Stage 01: Problem Framing & Implementation
# 
# **Objective:** Optimize diversified investment portfolios under fluctuating market conditions while managing risk exposure.
# 
# **Stakeholders:** Portfolio managers, institutional investors, financial analysts, and risk management teams.
# 
# **Deliverables:** 
# - Descriptive analytics (patterns, correlations, trends)
# - Predictive outputs (forecasted returns, VaR, Sharpe ratios)
# - Optimized allocation weights with constraints

# %% [markdown]
# ## 1. Setup and Dependencies

# %%
# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from scipy.stats import norm
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import json

# Configuration
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("✅ All dependencies loaded successfully!")

# %% [markdown]
# ## 2. Portfolio Optimizer Class Definition

# %%
class PortfolioOptimizer:
    """
    A comprehensive portfolio optimization system for institutional investors.
    Provides descriptive analytics, risk metrics, and optimal allocation recommendations.
    """
    
    def __init__(self, assets, start_date=None, end_date=None):
        """
        Initialize the portfolio optimizer.
        
        Args:
            assets (list): List of asset tickers (e.g., ['AAPL', 'GOOGL', 'MSFT'])
            start_date (str): Start date for data collection (YYYY-MM-DD)
            end_date (str): End date for data collection (YYYY-MM-DD)
        """
        self.assets = assets
        self.start_date = start_date or (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.prices = None
        self.returns = None
        self.mean_returns = None
        self.cov_matrix = None
        
        print(f"🚀 Portfolio Optimizer initialized with {len(assets)} assets")
        print(f"📅 Date range: {self.start_date} to {self.end_date}")
        
    def validate_data(self):
        """Validate loaded data quality and provide diagnostics."""
        if self.prices is None:
            print("❌ No price data loaded")
            return False
            
        print(f"📊 DATA VALIDATION REPORT")
        print(f"=" * 30)
        print(f"Assets: {list(self.prices.columns)}")
        print(f"Data points: {len(self.prices)}")
        print(f"Date range: {self.prices.index[0]} to {self.prices.index[-1]}")
        
        # Check for missing data
        missing_data = self.prices.isnull().sum()
        if missing_data.any():
            print(f"⚠️  Missing data points:")
            for asset, missing in missing_data.items():
                if missing > 0:
                    print(f"  {asset}: {missing} missing values")
        else:
            print(f"✅ No missing data found")
            
        # Check for zero prices
        zero_prices = (self.prices == 0).sum()
        if zero_prices.any():
            print(f"⚠️  Zero price points:")
            for asset, zeros in zero_prices.items():
                if zeros > 0:
                    print(f"  {asset}: {zeros} zero values")

        
                    
        return True
        
    def load_data(self):
        """Load historical price data for the assets."""
        try:
            print(f"📊 Loading data for: {', '.join(self.assets)}")
            
            # Download data with progress disabled for cleaner output
            data = yf.download(self.assets, start=self.start_date, end=self.end_date, progress=False)
            
            if data.empty:
                print("❌ No data returned from yfinance")
                return False
            
            # Priority order for price columns (prefer Adj Close, fallback to Close)
            price_columns = ['Adj Close', 'Close']
            selected_price_col = None
            
            # Handle different data structures based on number of assets
            if len(self.assets) == 1:
                # Single asset - data is a DataFrame with OHLCV columns
                for col in price_columns:
                    if col in data.columns:
                        selected_price_col = col
                        self.prices = data[[col]].copy()
                        self.prices.columns = self.assets
                        break
                
                if selected_price_col is None:
                    print("❌ Neither 'Adj Close' nor 'Close' found. Available columns:", data.columns.tolist())
                    return False
                    
            else:
                # Multiple assets - data has MultiIndex columns
                if isinstance(data.columns, pd.MultiIndex):
                    available_level0 = data.columns.get_level_values(0).unique()
                    
                    for col in price_columns:
                        if col in available_level0:
                            selected_price_col = col
                            self.prices = data[col].copy()
                            break
                    
                    if selected_price_col is None:
                        print("❌ Neither 'Adj Close' nor 'Close' found in MultiIndex.")
                        print("Available level 0:", available_level0.tolist())
                        return False
                else:
                    # Fallback - try to extract price columns
                    for col in price_columns:
                        price_cols = [c for c in data.columns if col in str(c)]
                        if price_cols:
                            selected_price_col = col
                            self.prices = data[price_cols].copy()
                            # Clean column names
                            self.prices.columns = [c.replace(col, '').strip() for c in self.prices.columns]
                            break
                    
                    if selected_price_col is None:
                        print("❌ Could not find price data. Available columns:", data.columns.tolist())
                        return False
            
            print(f"📊 Using '{selected_price_col}' price data")
            
            # Ensure we have a DataFrame
            if isinstance(self.prices, pd.Series):
                self.prices = self.prices.to_frame()
            
            # Handle missing data and validate
            initial_shape = self.prices.shape
            self.prices = self.prices.dropna()
            
            if self.prices.empty:
                print("❌ No data available after cleaning. Check date range and asset symbols.")
                return False
            
            if self.prices.shape[0] < 50:  # Need sufficient data for meaningful analysis
                print(f"⚠️  Warning: Only {self.prices.shape[0]} data points available. Consider extending date range.")
            
            # Validate that all assets have data
            if self.prices.shape[1] != len(self.assets):
                missing_assets = set(self.assets) - set(self.prices.columns)
                if missing_assets:
                    print(f"⚠️  Warning: No data for assets: {missing_assets}")
            
            print(f"✅ Data loaded successfully!")
            print(f"📈 Shape: {self.prices.shape} (removed {initial_shape[0] - self.prices.shape[0]} rows with missing data)")
            print(f"📅 Date range: {self.prices.index[0].strftime('%Y-%m-%d')} to {self.prices.index[-1].strftime('%Y-%m-%d')}")
            print(f"💼 Assets loaded: {list(self.prices.columns)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            print("🔍 Troubleshooting tips:")
            print("  1. Check internet connection")
            print("  2. Verify asset symbols are correct")
            print("  3. Try a different date range")
            print("  4. Check if markets were open during the specified period")
            return False
    
    def calculate_returns(self):
        """Calculate daily returns and key statistics."""
        if self.prices is None:
            raise ValueError("❌ Price data not loaded. Run load_data() first.")
        
        # Calculate daily returns
        self.returns = self.prices.pct_change().dropna()
        
        # Calculate annualized mean returns and covariance matrix
        self.mean_returns = self.returns.mean() * 252  # Annualized
        self.cov_matrix = self.returns.cov() * 252     # Annualized
        
        print("✅ Returns calculated and annualized")
        print(f"📊 Returns shape: {self.returns.shape}")
        return self.returns
    
    def descriptive_analytics(self):
        """Generate descriptive analytics and visualizations."""
        if self.returns is None:
            self.calculate_returns()
        
        # Summary statistics
        stats = pd.DataFrame({
            'Mean Return (Annual)': self.mean_returns,
            'Volatility (Annual)': np.sqrt(np.diag(self.cov_matrix)),
            'Sharpe Ratio': self.mean_returns / np.sqrt(np.diag(self.cov_matrix)),
            'Skewness': self.returns.skew(),
            'Kurtosis': self.returns.kurtosis()
        })
        
        print("📈 DESCRIPTIVE ANALYTICS")
        print("=" * 50)
        print(stats.round(4))
        
        # Correlation matrix
        corr_matrix = self.returns.corr()
        print(f"\n🔗 CORRELATION MATRIX")
        print("=" * 30)
        print(corr_matrix.round(3))
        
        return stats, corr_matrix
    
    def plot_analytics(self):
        """Create comprehensive visualization dashboard."""
        if self.returns is None:
            self.calculate_returns()
            
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Portfolio Analytics Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Price evolution
        self.prices.plot(ax=axes[0,0], linewidth=2)
        axes[0,0].set_title('📈 Asset Price Evolution', fontweight='bold')
        axes[0,0].set_ylabel('Price ($)')
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 2. Returns distribution
        self.returns.plot(kind='hist', bins=50, ax=axes[0,1], alpha=0.7, stacked=True)
        axes[0,1].set_title('📊 Returns Distribution', fontweight='bold')
        axes[0,1].set_xlabel('Daily Returns')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Correlation heatmap
        corr_matrix = self.returns.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', center=0, 
                   ax=axes[1,0], fmt='.2f', square=True)
        axes[1,0].set_title('🔗 Asset Correlation Matrix', fontweight='bold')
        
        # 4. Risk-Return scatter
        vol = np.sqrt(np.diag(self.cov_matrix))
        scatter = axes[1,1].scatter(vol, self.mean_returns, s=150, alpha=0.7, c=range(len(self.assets)), cmap='viridis')
        for i, asset in enumerate(self.assets):
            axes[1,1].annotate(asset, (vol[i], self.mean_returns[i]), 
                             xytext=(5, 5), textcoords='offset points', fontweight='bold')
        axes[1,1].set_xlabel('Volatility (Annual)')
        axes[1,1].set_ylabel('Expected Return (Annual)')
        axes[1,1].set_title('⚡ Risk-Return Profile', fontweight='bold')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def portfolio_metrics(self, weights):
        """Calculate portfolio performance metrics."""
        weights = np.array(weights)
        
        # Portfolio return and risk
        port_return = np.sum(self.mean_returns * weights)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        sharpe_ratio = port_return / port_vol if port_vol > 0 else 0
        
        return {
            'return': port_return,
            'volatility': port_vol,
            'sharpe_ratio': sharpe_ratio
        }
    
    def calculate_var(self, weights, confidence_level=0.05):
        """Calculate Value at Risk (VaR) for the portfolio."""
        weights = np.array(weights)
        
        # Portfolio returns
        port_returns = np.dot(self.returns, weights)
        
        # Historical VaR
        var_hist = np.percentile(port_returns, confidence_level * 100)
        
        # Parametric VaR
        port_mean = np.mean(port_returns)
        port_std = np.std(port_returns)
        var_param = norm.ppf(confidence_level, port_mean, port_std)
        
        return {
            'historical_var': var_hist,
            'parametric_var': var_param
        }
    
    def optimize_portfolio(self, objective='sharpe', constraints=None):
        """Optimize portfolio allocation."""
        # Check if data is loaded
        if self.prices is None or self.mean_returns is None or self.cov_matrix is None:
            print("❌ Cannot optimize: No data loaded or returns not calculated")
            print("💡 Please run load_data() and calculate_returns() first")
            return None
            
        n_assets = len(self.assets)
        
        # Objective functions
        def neg_sharpe(weights):
            metrics = self.portfolio_metrics(weights)
            return -metrics['sharpe_ratio']
        
        def portfolio_vol(weights):
            return self.portfolio_metrics(weights)['volatility']
        
        def neg_return(weights):
            return -self.portfolio_metrics(weights)['return']
        
        # Choose objective
        objective_functions = {
            'sharpe': neg_sharpe,
            'min_vol': portfolio_vol,
            'max_return': neg_return
        }
        
        obj_func = objective_functions[objective]
        
        # Constraints
        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # Weights sum to 1
        
        if constraints:
            if 'max_weight' in constraints:
                for i in range(n_assets):
                    cons.append({'type': 'ineq', 
                               'fun': lambda x, i=i: constraints['max_weight'] - x[i]})
            
            if 'min_weight' in constraints:
                for i in range(n_assets):
                    cons.append({'type': 'ineq', 
                               'fun': lambda x, i=i: x[i] - constraints['min_weight']})
        
        # Bounds (0 to 1 for each weight)
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        # Initial guess (equal weights)
        x0 = np.array([1/n_assets] * n_assets)
        
        # Optimize
        try:
            result = minimize(obj_func, x0, method='SLSQP', bounds=bounds, constraints=cons)
            
            if result.success:
                optimal_weights = result.x
                metrics = self.portfolio_metrics(optimal_weights)
                var_metrics = self.calculate_var(optimal_weights)
                
                return {
                    'weights': optimal_weights,
                    'metrics': metrics,
                    'var': var_metrics,
                    'optimization_result': result
                }
            else:
                print(f"❌ Optimization failed: {result.message}")
                return None
        except Exception as e:
            print(f"❌ Optimization error: {e}")
            return None
    
    def efficient_frontier(self, n_portfolios=100):
        """Generate the efficient frontier."""
        n_assets = len(self.assets)
        results = np.zeros((3, n_portfolios))
        
        # Generate target returns
        min_ret = self.mean_returns.min()
        max_ret = self.mean_returns.max()
        target_returns = np.linspace(min_ret, max_ret, n_portfolios)
        
        def portfolio_vol(weights):
            return self.portfolio_metrics(weights)['volatility']
        
        # Constraints
        cons = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Weights sum to 1
        ]
        
        bounds = tuple((0, 1) for _ in range(n_assets))
        x0 = np.array([1/n_assets] * n_assets)
        
        print("🔄 Computing efficient frontier...")
        
        for i, target_ret in enumerate(target_returns):
            # Add return constraint
            cons_with_return = cons + [
                {'type': 'eq', 'fun': lambda x, target=target_ret: 
                 np.sum(self.mean_returns * x) - target}
            ]
            
            result = minimize(portfolio_vol, x0, method='SLSQP', 
                            bounds=bounds, constraints=cons_with_return)
            
            if result.success:
                weights = result.x
                metrics = self.portfolio_metrics(weights)
                results[0, i] = metrics['return']
                results[1, i] = metrics['volatility']
                results[2, i] = metrics['sharpe_ratio']
        
        print("✅ Efficient frontier computed!")
        return results[0], results[1], results[2]
    
    def plot_efficient_frontier(self):
        """Plot the efficient frontier with optimal portfolios."""
        returns, volatilities, sharpe_ratios = self.efficient_frontier()
        
        plt.figure(figsize=(12, 8))
        
        # Plot efficient frontier
        plt.plot(volatilities, returns, 'b-', linewidth=3, label='Efficient Frontier', alpha=0.8)
        
        # Plot individual assets
        asset_vol = np.sqrt(np.diag(self.cov_matrix))
        plt.scatter(asset_vol, self.mean_returns, s=150, c='red', marker='o', 
                   alpha=0.7, edgecolors='darkred', linewidth=2, label='Individual Assets')
        for i, asset in enumerate(self.assets):
            plt.annotate(asset, (asset_vol[i], self.mean_returns[i]), 
                        xytext=(8, 8), textcoords='offset points', fontweight='bold')
        
        # Plot optimal portfolios
        max_sharpe_opt = self.optimize_portfolio('sharpe')
        min_vol_opt = self.optimize_portfolio('min_vol')
        
        if max_sharpe_opt:
            plt.scatter(max_sharpe_opt['metrics']['volatility'], 
                       max_sharpe_opt['metrics']['return'], 
                       s=300, c='green', marker='*', label='Max Sharpe', 
                       edgecolors='darkgreen', linewidth=2)
        
        if min_vol_opt:
            plt.scatter(min_vol_opt['metrics']['volatility'], 
                       min_vol_opt['metrics']['return'], 
                       s=300, c='orange', marker='*', label='Min Volatility',
                       edgecolors='darkorange', linewidth=2)
        
        plt.xlabel('Volatility (Annual)', fontweight='bold')
        plt.ylabel('Expected Return (Annual)', fontweight='bold')
        plt.title('🎯 Efficient Frontier with Optimal Portfolios', fontsize=14, fontweight='bold')
        plt.legend(frameon=True, fancybox=True, shadow=True)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

print("✅ PortfolioOptimizer class defined successfully!")

# %% [markdown]
# ## 3. Portfolio Configuration
# 
# Define your assets and parameters here. You can modify these based on your specific requirements.

# %%
# Define portfolio assets
# Modify this list based on your specific portfolio
ASSETS = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']  # Tech-focused portfolio
# ASSETS = ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM']       # ETF diversified portfolio
# ASSETS = ['AAPL', 'JPM', 'JNJ', 'PG', 'XOM']       # Mixed sectors portfolio

# Date range for analysis (you can modify these)
START_DATE = '2020-01-01'
END_DATE = None  # None means today

print(f"🎯 Portfolio Configuration:")
print(f"Assets: {', '.join(ASSETS)}")
print(f"Start Date: {START_DATE}")
print(f"End Date: {'Today' if END_DATE is None else END_DATE}")

# %% [markdown]
# ## 4. Initialize Portfolio Optimizer

# %%
# Initialize the portfolio optimizer
optimizer = PortfolioOptimizer(ASSETS, start_date=START_DATE, end_date=END_DATE)

# Load market data
success = optimizer.load_data()

if success:
    # Validate the data quality
    optimizer.validate_data()
    print("🎉 Portfolio optimizer ready for analysis!")
else:
    print("❌ Failed to load data. Please check your asset symbols and internet connection.")
    print("💡 Try these troubleshooting steps:")
    print("1. Check if asset symbols are correct (e.g., 'AAPL' not 'Apple')")
    print("2. Verify internet connection")
    print("3. Try a more recent start date")
    print("4. Test with a single asset first: ASSETS = ['AAPL']")

# %% [markdown]
# ## 5. Descriptive Analytics
# 
# This section provides comprehensive descriptive analytics including correlations, risk metrics, and visualizations.

# %%
# Generate descriptive analytics
if optimizer.prices is not None:
    stats, correlation_matrix = optimizer.descriptive_analytics()
else:
    print("❌ Cannot run analytics - no data loaded. Please run the data loading cell above first.")
    print("💡 Make sure the previous cell executed successfully before running this one.")

# %% 
# Create visualization dashboard
if optimizer.prices is not None:
    optimizer.plot_analytics()
else:
    print("❌ Cannot create visualizations - no data loaded.")

# %% [markdown]
# ## 6. Portfolio Optimization
# 
# Now we'll optimize the portfolio using different strategies and constraints.

# %% [markdown]
# ### 6.1 Maximum Sharpe Ratio Portfolio

# %%
print("🎯 MAXIMUM SHARPE RATIO OPTIMIZATION")
print("=" * 50)

# Ensure data is loaded and processed
if optimizer.prices is not None:
    # Make sure returns are calculated
    if optimizer.mean_returns is None:
        optimizer.calculate_returns()
    
    max_sharpe_result = optimizer.optimize_portfolio('sharpe')

    if max_sharpe_result:
        print("✅ Optimization successful!")
        print(f"\n📊 OPTIMAL WEIGHTS:")
        weights_df = pd.DataFrame({
            'Asset': ASSETS,
            'Weight': max_sharpe_result['weights'],
            'Weight (%)': max_sharpe_result['weights'] * 100
        }).round(4)
        print(weights_df)
        
        metrics = max_sharpe_result['metrics']
        print(f"\n📈 PORTFOLIO METRICS:")
        print(f"Expected Annual Return: {metrics['return']:.3f} ({metrics['return']*100:.1f}%)")
        print(f"Annual Volatility: {metrics['volatility']:.3f} ({metrics['volatility']*100:.1f}%)")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
        
        var_metrics = max_sharpe_result['var']
        print(f"\n⚠️  RISK METRICS (VaR at 95% confidence):")
        print(f"Historical VaR: {var_metrics['historical_var']:.4f} ({var_metrics['historical_var']*100:.2f}%)")
        print(f"Parametric VaR: {var_metrics['parametric_var']:.4f} ({var_metrics['parametric_var']*100:.2f}%)")
    else:
        print("❌ Optimization failed!")
else:
    print("❌ Cannot run optimization - no data loaded")
    print("💡 Please ensure the data loading step completed successfully")

# %% [markdown]
# ### 6.2 Minimum Volatility Portfolio

# %%
print("🛡️  MINIMUM VOLATILITY OPTIMIZATION")
print("=" * 50)

if optimizer.prices is not None and optimizer.mean_returns is not None:
    min_vol_result = optimizer.optimize_portfolio('min_vol')

    if min_vol_result:
        print("✅ Optimization successful!")
        print(f"\n📊 OPTIMAL WEIGHTS:")
        weights_df = pd.DataFrame({
            'Asset': ASSETS,
            'Weight': min_vol_result['weights'],
            'Weight (%)': min_vol_result['weights'] * 100
        }).round(4)
        print(weights_df)
        
        metrics = min_vol_result['metrics']
        print(f"\n📈 PORTFOLIO METRICS:")
        print(f"Expected Annual Return: {metrics['return']:.3f} ({metrics['return']*100:.1f}%)")
        print(f"Annual Volatility: {metrics['volatility']:.3f} ({metrics['volatility']*100:.1f}%)")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
        
        var_metrics = min_vol_result['var']
        print(f"\n⚠️  RISK METRICS (VaR at 95% confidence):")
        print(f"Historical VaR: {var_metrics['historical_var']:.4f} ({var_metrics['historical_var']*100:.2f}%)")
        print(f"Parametric VaR: {var_metrics['parametric_var']:.4f} ({var_metrics['parametric_var']*100:.2f}%)")
    else:
        print("❌ Optimization failed!")
else:
    print("❌ Cannot run optimization - no data loaded")

# %% [markdown]
# ### 6.3 Constrained Portfolio Optimization
# 
# This demonstrates how to add regulatory or risk management constraints.

# %%
print("📋 CONSTRAINED PORTFOLIO OPTIMIZATION")
print("=" * 50)
print("Constraints: Maximum 30% per asset, Minimum 5% per asset")

# Define constraints (typical for institutional portfolios)
constraints = {
    'max_weight': 0.30,  # Maximum 30% in any single asset
    'min_weight': 0.05   # Minimum 5% in each asset
}

constrained_result = optimizer.optimize_portfolio('sharpe', constraints=constraints)

if constrained_result:
    print("✅ Constrained optimization successful!")
    print(f"\n📊 CONSTRAINED OPTIMAL WEIGHTS:")
    weights_df = pd.DataFrame({
        'Asset': ASSETS,
        'Weight': constrained_result['weights'],
        'Weight (%)': constrained_result['weights'] * 100
    }).round(4)
    display(weights_df)
    
    metrics = constrained_result['metrics']
    print(f"\n📈 PORTFOLIO METRICS:")
    print(f"Expected Annual Return: {metrics['return']:.3f} ({metrics['return']*100:.1f}%)")
    print(f"Annual Volatility: {metrics['volatility']:.3f} ({metrics['volatility']*100:.1f}%)")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
else:
    print("❌ Constrained optimization failed!")

# %% [markdown]
# ## 7. Efficient Frontier Analysis

# %%
# Plot the efficient frontier
optimizer.plot_efficient_frontier()

# %% [markdown]
# ## 8. Portfolio Comparison & Results Summary

# %%
print("📊 PORTFOLIO OPTIMIZATION RESULTS SUMMARY")
print("=" * 60)

# Create comparison table
results_data = []

if max_sharpe_result:
    results_data.append({
        'Strategy': 'Max Sharpe Ratio',
        'Return': f"{max_sharpe_result['metrics']['return']:.3f}",
        'Volatility': f"{max_sharpe_result['metrics']['volatility']:.3f}",
        'Sharpe Ratio': f"{max_sharpe_result['metrics']['sharpe_ratio']:.3f}",
        'VaR (5%)': f"{max_sharpe_result['var']['historical_var']:.4f}"
    })

if min_vol_result:
    results_data.append({
        'Strategy': 'Min Volatility',
        'Return': f"{min_vol_result['metrics']['return']:.3f}",
        'Volatility': f"{min_vol_result['metrics']['volatility']:.3f}",
        'Sharpe Ratio': f"{min_vol_result['metrics']['sharpe_ratio']:.3f}",
        'VaR (5%)': f"{min_vol_result['var']['historical_var']:.4f}"
    })

if constrained_result:
    results_data.append({
        'Strategy': 'Constrained (Max 30%)',
        'Return': f"{constrained_result['metrics']['return']:.3f}",
        'Volatility': f"{constrained_result['metrics']['volatility']:.3f}",
        'Sharpe Ratio': f"{constrained_result['metrics']['sharpe_ratio']:.3f}",
        'VaR (5%)': f"{constrained_result['var']['historical_var']:.4f}"
    })

if results_data:
    results_df = pd.DataFrame(results_data)
    print(results_df)

print(f"\n✅ Analysis completed for portfolio: {', '.join(ASSETS)}")
print(f"📅 Data period: {START_DATE} to {optimizer.end_date}")

# %% [markdown]
# ## 9. Export Results (Optional)
# 
# Save your optimization results for stakeholder presentations.

# %%
# Uncomment to save results to JSON file
# save_path = f"portfolio_optimization_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
# 
# report_data = {
#     'analysis_date': datetime.now().isoformat(),
#     'assets': ASSETS,
#     'date_range': {'start': START_DATE, 'end': optimizer.end_date},
#     'statistics': stats.to_dict(),
#     'correlation_matrix': correlation_matrix.to_dict(),
#     'optimization_results': {}
# }
# 
# if max_sharpe_result:
#     report_data['optimization_results']['max_sharpe'] = {
#         'weights': max_sharpe_result['weights'].tolist(),
#         'metrics': max_sharpe_result['metrics'],
#         'var': max_sharpe_result['var']
#     }
# 
# if min_vol_result:
#     report_data['optimization_results']['min_volatility'] = {
#         'weights': min_vol_result['weights'].tolist(),
#         'metrics': min_vol_result['metrics'],
#         'var': min_vol_result['var']
#     }
# 
# with open(save_path, 'w') as f:
#     json.dump(report_data, f, indent=2, default=str)
# 
# print(f"📁 Results saved to: {save_path}")

# %% [markdown]
# ## 10. Next Steps & Recommendations
# 
# ### For Stakeholders:
# 
# 1. **Portfolio Review**: Use the optimization results during your next portfolio review cycle
# 2. **Risk Assessment**: Pay attention to the VaR metrics for risk management
# 3. **Constraint Validation**: Ensure the constrained portfolio meets your regulatory requirements
# 4. **Backtesting**: Consider implementing backtesting for the recommended allocations
# 
# ### For Further Development:
# 
# 1. **Scenario Analysis**: Add stress testing under different market conditions
# 2. **Dynamic Rebalancing**: Implement time-varying optimization
# 3. **Alternative Risk Measures**: Consider CVaR, Maximum Drawdown
# 4. **Factor Models**: Integrate factor-based risk models
# 5. **Real-time Updates**: Connect to live market data feeds
# 
# ### Model Limitations & Risk Mitigation:
# 
# - **Historical Bias**: Models assume past patterns continue
# - **Correlation Instability**: Asset correlations change during crises  
# - **Black Swan Events**: Extreme events may not be captured
# 
# **Mitigation Strategies**: Regular reoptimization, scenario stress testing, and maintaining some portfolio flexibility.