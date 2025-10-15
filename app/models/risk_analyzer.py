import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
from scipy import stats
from sklearn.covariance import EmpiricalCovariance
import warnings

warnings.filterwarnings('ignore')

class RiskAnalyzer:
    """Comprehensive financial risk analysis and portfolio risk management"""
    
    def __init__(self):
        self.risk_free_rate = 0.02  # 2% default risk-free rate
        self.confidence_levels = [0.95, 0.99]  # For VaR calculations
        
        # Risk categories and thresholds
        self.risk_thresholds = {
            'volatility': {'low': 0.15, 'medium': 0.25, 'high': 0.40},
            'beta': {'low': 0.8, 'medium': 1.2, 'high': 1.5},
            'sharpe': {'excellent': 2.0, 'good': 1.0, 'poor': 0.5},
            'max_drawdown': {'low': 0.05, 'medium': 0.15, 'high': 0.25}
        }
    
    def _fetch_market_data(self, symbols: List[str], period: str = "2y") -> Dict[str, pd.DataFrame]:
        """Fetch market data for multiple symbols"""
        market_data = {}
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(period=period)
                if not df.empty:
                    market_data[symbol] = df
                else:
                    print(f"Warning: No data found for {symbol}")
            except Exception as e:
                print(f"Error fetching data for {symbol}: {str(e)}")
        
        return market_data
    
    def _get_benchmark_data(self, period: str = "2y") -> pd.DataFrame:
        """Get benchmark data (S&P 500)"""
        spy = yf.Ticker("SPY")
        return spy.history(period=period)
    
    def calculate_basic_risk_metrics(self, symbol: str, period: str = "1y") -> Dict:
        """Calculate basic risk metrics for a single asset"""
        
        try:
            market_data = self._fetch_market_data([symbol], period)
            if symbol not in market_data:
                raise ValueError(f"No data available for {symbol}")
            
            df = market_data[symbol]
            prices = df['Close']
            returns = prices.pct_change().dropna()
            
            # Get benchmark data
            benchmark_df = self._get_benchmark_data(period)
            benchmark_returns = benchmark_df['Close'].pct_change().dropna()
            
            # Align dates
            common_dates = returns.index.intersection(benchmark_returns.index)
            returns_aligned = returns[common_dates]
            benchmark_aligned = benchmark_returns[common_dates]
            
            # Basic risk metrics
            volatility = returns_aligned.std() * np.sqrt(252)  # Annualized
            mean_return = returns_aligned.mean() * 252  # Annualized
            
            # Beta calculation
            if len(benchmark_aligned) > 0:
                covariance = np.cov(returns_aligned, benchmark_aligned)[0][1]
                benchmark_variance = benchmark_aligned.var()
                beta = covariance / benchmark_variance if benchmark_variance != 0 else 1.0
            else:
                beta = 1.0
            
            # Sharpe Ratio
            sharpe_ratio = (mean_return - self.risk_free_rate) / volatility if volatility != 0 else 0
            
            # Maximum Drawdown
            cumulative_returns = (1 + returns_aligned).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            # Value at Risk (VaR)
            var_95 = np.percentile(returns_aligned, 5)
            var_99 = np.percentile(returns_aligned, 1)
            
            # Conditional VaR (Expected Shortfall)
            cvar_95 = returns_aligned[returns_aligned <= var_95].mean()
            cvar_99 = returns_aligned[returns_aligned <= var_99].mean()
            
            # Risk categorization
            risk_level = self._categorize_risk_level({
                'volatility': volatility,
                'beta': beta,
                'sharpe': sharpe_ratio,
                'max_drawdown': abs(max_drawdown)
            })
            
            return {
                'symbol': symbol,
                'volatility': float(volatility),
                'beta': float(beta),
                'sharpe_ratio': float(sharpe_ratio),
                'max_drawdown': float(max_drawdown),
                'var_95': float(var_95),
                'var_99': float(var_99),
                'cvar_95': float(cvar_95),
                'cvar_99': float(cvar_99),
                'mean_return': float(mean_return),
                'risk_level': risk_level,
                'risk_score': self._calculate_risk_score(volatility, beta, sharpe_ratio, abs(max_drawdown)),
                'calculation_date': datetime.now().isoformat()
            }
            
        except Exception as e:
            raise Exception(f"Error calculating risk metrics for {symbol}: {str(e)}")
    
    def _categorize_risk_level(self, metrics: Dict) -> str:
        """Categorize overall risk level based on multiple metrics"""
        
        risk_scores = []
        
        # Volatility score
        if metrics['volatility'] <= self.risk_thresholds['volatility']['low']:
            risk_scores.append(1)  # Low risk
        elif metrics['volatility'] <= self.risk_thresholds['volatility']['medium']:
            risk_scores.append(2)  # Medium risk
        else:
            risk_scores.append(3)  # High risk
        
        # Beta score
        if abs(metrics['beta'] - 1.0) <= 0.2:
            risk_scores.append(2)  # Market risk
        elif metrics['beta'] > self.risk_thresholds['beta']['high']:
            risk_scores.append(3)  # High risk
        else:
            risk_scores.append(1)  # Low risk
        
        # Sharpe ratio score (inverted - higher is better)
        if metrics['sharpe'] >= self.risk_thresholds['sharpe']['excellent']:
            risk_scores.append(1)  # Low risk (good performance)
        elif metrics['sharpe'] >= self.risk_thresholds['sharpe']['good']:
            risk_scores.append(2)  # Medium risk
        else:
            risk_scores.append(3)  # High risk
        
        # Max drawdown score
        if metrics['max_drawdown'] <= self.risk_thresholds['max_drawdown']['low']:
            risk_scores.append(1)
        elif metrics['max_drawdown'] <= self.risk_thresholds['max_drawdown']['medium']:
            risk_scores.append(2)
        else:
            risk_scores.append(3)
        
        # Overall risk level
        avg_score = np.mean(risk_scores)
        
        if avg_score <= 1.5:
            return "Low"
        elif avg_score <= 2.5:
            return "Medium"
        else:
            return "High"
    
    def _calculate_risk_score(self, volatility: float, beta: float, 
                            sharpe: float, max_drawdown: float) -> float:
        """Calculate a composite risk score (0-10 scale)"""
        
        # Normalize metrics to 0-10 scale
        vol_score = min(volatility * 20, 10)  # Cap at 50% volatility = 10
        beta_score = min(abs(beta - 1.0) * 5, 10)  # Deviation from market
        sharpe_score = max(0, min(10, (2 - sharpe) * 2.5))  # Inverted (lower Sharpe = higher risk)
        drawdown_score = min(max_drawdown * 40, 10)  # 25% drawdown = 10
        
        # Weighted composite score
        risk_score = (
            vol_score * 0.3 +
            beta_score * 0.2 +
            sharpe_score * 0.3 +
            drawdown_score * 0.2
        )
        
        return float(min(risk_score, 10.0))
    
    def analyze_portfolio_risk(self, portfolio: Dict[str, float], period: str = "1y") -> Dict:
        """Analyze risk metrics for an entire portfolio"""
        
        symbols = list(portfolio.keys())
        weights = np.array(list(portfolio.values()))
        
        # Validate weights
        if abs(weights.sum() - 1.0) > 1e-6:
            weights = weights / weights.sum()  # Normalize weights
        
        # Fetch market data
        market_data = self._fetch_market_data(symbols, period)
        
        if not market_data:
            raise ValueError("No market data available for portfolio analysis")
        
        # Calculate returns matrix
        returns_data = {}
        for symbol in symbols:
            if symbol in market_data:
                returns = market_data[symbol]['Close'].pct_change().dropna()
                returns_data[symbol] = returns
        
        if not returns_data:
            raise ValueError("No valid return data for portfolio analysis")
        
        # Align all return series
        returns_df = pd.DataFrame(returns_data)
        returns_df = returns_df.dropna()
        
        # Portfolio returns
        portfolio_returns = (returns_df * weights).sum(axis=1)
        
        # Portfolio risk metrics
        portfolio_volatility = portfolio_returns.std() * np.sqrt(252)
        portfolio_mean_return = portfolio_returns.mean() * 252
        
        # Portfolio Sharpe ratio
        portfolio_sharpe = (portfolio_mean_return - self.risk_free_rate) / portfolio_volatility if portfolio_volatility != 0 else 0
        
        # Portfolio VaR
        portfolio_var_95 = np.percentile(portfolio_returns, 5)
        portfolio_var_99 = np.percentile(portfolio_returns, 1)
        
        # Correlation matrix
        correlation_matrix = returns_df.corr().to_dict()
        
        # Portfolio beta (vs benchmark)
        benchmark_df = self._get_benchmark_data(period)
        benchmark_returns = benchmark_df['Close'].pct_change().dropna()
        
        # Align with portfolio returns
        common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
        portfolio_aligned = portfolio_returns[common_dates]
        benchmark_aligned = benchmark_returns[common_dates]
        
        if len(benchmark_aligned) > 0:
            portfolio_beta = np.cov(portfolio_aligned, benchmark_aligned)[0][1] / benchmark_aligned.var()
        else:
            portfolio_beta = 1.0
        
        # Individual asset analysis
        individual_risks = {}
        for symbol in symbols:
            if symbol in market_data:
                try:
                    risk_metrics = self.calculate_basic_risk_metrics(symbol, period)
                    individual_risks[symbol] = risk_metrics
                except Exception as e:
                    print(f"Warning: Could not calculate risk for {symbol}: {str(e)}")
        
        # Diversification ratio
        individual_volatilities = []
        for symbol in symbols:
            if symbol in returns_data:
                vol = returns_data[symbol].std() * np.sqrt(252)
                individual_volatilities.append(vol)
        
        if individual_volatilities:
            weighted_avg_volatility = np.average(individual_volatilities, weights=weights[:len(individual_volatilities)])
            diversification_ratio = weighted_avg_volatility / portfolio_volatility if portfolio_volatility != 0 else 1.0
        else:
            diversification_ratio = 1.0
        
        # Portfolio risk score
        portfolio_risk_score = self._calculate_risk_score(
            portfolio_volatility, portfolio_beta, portfolio_sharpe, 0.15  # Assume moderate drawdown
        )
        
        # Risk recommendations
        recommendations = self._generate_risk_recommendations(
            portfolio_volatility, portfolio_beta, portfolio_sharpe, correlation_matrix
        )
        
        return {
            'portfolio_metrics': {
                'volatility': float(portfolio_volatility),
                'mean_return': float(portfolio_mean_return),
                'sharpe_ratio': float(portfolio_sharpe),
                'beta': float(portfolio_beta),
                'var_95': float(portfolio_var_95),
                'var_99': float(portfolio_var_99),
                'diversification_ratio': float(diversification_ratio),
                'risk_score': float(portfolio_risk_score)
            },
            'individual_risks': individual_risks,
            'correlation_matrix': correlation_matrix,
            'portfolio_composition': dict(zip(symbols, weights.tolist())),
            'risk_recommendations': recommendations,
            'analysis_date': datetime.now().isoformat()
        }
    
    def _generate_risk_recommendations(self, volatility: float, beta: float, 
                                     sharpe: float, correlation_matrix: Dict) -> List[str]:
        """Generate actionable risk management recommendations"""
        
        recommendations = []
        
        # Volatility recommendations
        if volatility > self.risk_thresholds['volatility']['high']:
            recommendations.append(
                "High portfolio volatility detected. Consider adding low-volatility assets or bonds to reduce risk."
            )
        elif volatility < self.risk_thresholds['volatility']['low']:
            recommendations.append(
                "Low portfolio volatility. You may be able to increase expected returns by adding growth assets."
            )
        
        # Beta recommendations
        if beta > self.risk_thresholds['beta']['high']:
            recommendations.append(
                f"Portfolio beta ({beta:.2f}) is high. Consider defensive stocks or market-neutral strategies during volatile periods."
            )
        elif beta < 0.8:
            recommendations.append(
                f"Portfolio beta ({beta:.2f}) is low. Portfolio may underperform in bull markets."
            )
        
        # Sharpe ratio recommendations
        if sharpe < self.risk_thresholds['sharpe']['poor']:
            recommendations.append(
                "Low risk-adjusted returns (Sharpe ratio). Review asset allocation and consider higher-quality investments."
            )
        
        # Correlation recommendations
        if correlation_matrix:
            correlations = []
            symbols = list(correlation_matrix.keys())
            for i, symbol1 in enumerate(symbols):
                for j, symbol2 in enumerate(symbols[i+1:], i+1):
                    if symbol1 in correlation_matrix and symbol2 in correlation_matrix[symbol1]:
                        corr = correlation_matrix[symbol1][symbol2]
                        if not np.isnan(corr):
                            correlations.append(abs(corr))
            
            if correlations:
                avg_correlation = np.mean(correlations)
                if avg_correlation > 0.8:
                    recommendations.append(
                        "High correlation between assets reduces diversification benefits. Consider assets from different sectors or asset classes."
                    )
        
        if not recommendations:
            recommendations.append("Portfolio risk profile appears well-balanced. Continue monitoring market conditions.")
        
        return recommendations
    
    def stress_test_portfolio(self, portfolio: Dict[str, float], 
                            stress_scenarios: Dict[str, float] = None) -> Dict:
        """Perform stress testing on portfolio under various market scenarios"""
        
        if stress_scenarios is None:
            stress_scenarios = {
                'market_crash': -0.20,     # 20% market decline
                'mild_correction': -0.10,   # 10% market decline
                'volatility_spike': 0.5,    # 50% increase in volatility
                'interest_rate_shock': 0.02 # 200 bps rate increase
            }
        
        symbols = list(portfolio.keys())
        weights = np.array(list(portfolio.values()))
        
        # Fetch market data
        market_data = self._fetch_market_data(symbols, period="1y")
        
        # Calculate current portfolio value (assume $100,000 initial)
        base_value = 100000
        
        stress_results = {}
        
        for scenario_name, shock_magnitude in stress_scenarios.items():
            scenario_results = {}
            
            if scenario_name in ['market_crash', 'mild_correction']:
                # Apply uniform shock to all assets
                shocked_returns = []
                for symbol in symbols:
                    if symbol in market_data:
                        current_price = market_data[symbol]['Close'].iloc[-1]
                        shocked_price = current_price * (1 + shock_magnitude)
                        shocked_return = (shocked_price - current_price) / current_price
                        shocked_returns.append(shocked_return)
                    else:
                        shocked_returns.append(shock_magnitude)
                
                portfolio_shocked_return = np.dot(weights, shocked_returns)
                portfolio_shocked_value = base_value * (1 + portfolio_shocked_return)
                
                scenario_results = {
                    'portfolio_return': float(portfolio_shocked_return),
                    'portfolio_value': float(portfolio_shocked_value),
                    'value_change': float(portfolio_shocked_value - base_value),
                    'individual_impacts': dict(zip(symbols, shocked_returns))
                }
            
            stress_results[scenario_name] = scenario_results
        
        return {
            'base_portfolio_value': base_value,
            'stress_scenarios': stress_results,
            'worst_case_scenario': min(stress_results.keys(), 
                                     key=lambda x: stress_results[x].get('portfolio_value', base_value)),
            'stress_test_date': datetime.now().isoformat()
        }
    
    def calculate_position_sizing(self, symbol: str, account_value: float, 
                                risk_tolerance: float = 0.02) -> Dict:
        """Calculate optimal position size based on risk management principles"""
        
        try:
            risk_metrics = self.calculate_basic_risk_metrics(symbol)
            volatility = risk_metrics['volatility']
            
            # Kelly Criterion approach (simplified)
            # Assume expected return slightly above risk-free rate
            expected_return = self.risk_free_rate + 0.05  # 5% risk premium
            
            if volatility > 0:
                kelly_fraction = (expected_return - self.risk_free_rate) / (volatility ** 2)
                kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
            else:
                kelly_fraction = 0.1  # Default 10%
            
            # Risk-based position sizing
            # Risk no more than specified percentage of account on single position
            max_position_risk = account_value * risk_tolerance
            
            # Estimate position size based on volatility
            if volatility > 0:
                volatility_based_size = max_position_risk / (volatility * account_value)
            else:
                volatility_based_size = 0.1
            
            # Take conservative approach
            recommended_fraction = min(kelly_fraction, volatility_based_size, 0.20)
            recommended_value = account_value * recommended_fraction
            
            return {
                'symbol': symbol,
                'account_value': account_value,
                'recommended_position_fraction': float(recommended_fraction),
                'recommended_position_value': float(recommended_value),
                'kelly_fraction': float(kelly_fraction),
                'volatility_based_fraction': float(volatility_based_size),
                'risk_metrics': risk_metrics,
                'risk_tolerance_used': risk_tolerance,
                'max_recommended_loss': float(max_position_risk)
            }
            
        except Exception as e:
            raise Exception(f"Error calculating position sizing for {symbol}: {str(e)}")