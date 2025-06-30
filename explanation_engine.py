import random
from datetime import datetime

class ExplanationEngine:
    """Generates natural language explanations for predictions"""
    
    def __init__(self):
        # Templates for different types of explanations
        self.bullish_templates = [
            "The stock shows strong bullish momentum with {indicators}. This suggests potential upward movement.",
            "Technical analysis indicates a positive trend forming with {indicators}, pointing towards a possible price increase.",
            "Multiple bullish signals detected: {indicators}. The stock appears ready for an upward move.",
            "Strong buying pressure evident from {indicators}, suggesting the stock may rise in the near term."
        ]
        
        self.bearish_templates = [
            "The technical indicators show bearish signals with {indicators}, suggesting potential downward pressure.",
            "Warning signs are emerging with {indicators}, indicating the stock might face selling pressure.",
            "Multiple bearish indicators detected: {indicators}. Consider caution as the stock may decline.",
            "The technical setup shows {indicators}, which typically precedes a price decline."
        ]
        
        self.neutral_templates = [
            "The stock is in a consolidation phase with {indicators}. Price movement may be limited in the short term.",
            "Mixed signals detected with {indicators}. The stock appears to be range-bound currently.",
            "Technical indicators show {indicators}, suggesting sideways movement or indecision in the market.",
            "The analysis reveals {indicators}, indicating the stock may sustain current levels."
        ]
    
    def generate_explanation(self, prediction_result, df, patterns, education_mode=True):
        """
        Generate a natural language explanation for the prediction
        
        Args:
            prediction_result (dict): ML prediction results
            df (pandas.DataFrame): Data with technical indicators
            patterns (list): Detected patterns
            education_mode (bool): Whether to include educational content
        
        Returns:
            str: Natural language explanation
        """
        try:
            prediction = prediction_result.get('prediction', 'SUSTAIN')
            confidence = prediction_result.get('confidence', 0.5)
            action = prediction_result.get('action', 'HOLD')
            
            # Analyze technical indicators
            indicators_analysis = self._analyze_indicators(df)
            
            # Choose appropriate template
            if prediction == 'UP':
                template = random.choice(self.bullish_templates)
            elif prediction == 'DOWN':
                template = random.choice(self.bearish_templates)
            else:
                template = random.choice(self.neutral_templates)
            
            # Format indicators text
            indicators_text = self._format_indicators_text(indicators_analysis)
            
            # Base explanation
            explanation = template.format(indicators=indicators_text)
            
            # Add confidence information
            confidence_text = self._get_confidence_text(confidence)
            explanation += f" {confidence_text}"
            
            # Add pattern information
            if patterns:
                pattern_text = self._format_patterns_text(patterns[-3:])  # Last 3 patterns
                explanation += f" Additionally, {pattern_text}"
            
            # Add action recommendation
            action_text = self._get_action_text(action, prediction)
            explanation += f" {action_text}"
            
            # Add educational content if requested
            if education_mode:
                educational_text = self._get_educational_text(prediction, indicators_analysis)
                explanation += f"\n\n📚 Learning Note: {educational_text}"
            
            return explanation
            
        except Exception as e:
            return f"Analysis complete. Prediction: {prediction_result.get('prediction', 'SUSTAIN')} with {prediction_result.get('confidence', 0.5):.1%} confidence."
    
    def _analyze_indicators(self, df):
        """Analyze technical indicators and return key insights"""
        if df is None or df.empty:
            return {}
        
        analysis = {}
        latest = df.iloc[-1]
        
        try:
            # RSI analysis
            if 'RSI' in df.columns:
                rsi = latest.get('RSI', 50)
                if rsi > 70:
                    analysis['RSI'] = 'overbought'
                elif rsi < 30:
                    analysis['RSI'] = 'oversold'
                elif rsi > 55:
                    analysis['RSI'] = 'bullish'
                elif rsi < 45:
                    analysis['RSI'] = 'bearish'
                else:
                    analysis['RSI'] = 'neutral'
            
            # MACD analysis
            if 'MACD' in df.columns and 'MACD_signal' in df.columns:
                macd = latest.get('MACD', 0)
                macd_signal = latest.get('MACD_signal', 0)
                if macd > macd_signal:
                    analysis['MACD'] = 'bullish crossover'
                else:
                    analysis['MACD'] = 'bearish crossover'
            
            # Bollinger Bands analysis
            if 'BB_percent' in df.columns:
                bb_percent = latest.get('BB_percent', 0.5)
                if bb_percent > 0.8:
                    analysis['BB'] = 'near upper band'
                elif bb_percent < 0.2:
                    analysis['BB'] = 'near lower band'
                else:
                    analysis['BB'] = 'within normal range'
            
            # Volume analysis
            if 'Volume' in df.columns and len(df) > 1:
                current_volume = latest.get('Volume', 0)
                avg_volume = df['Volume'].tail(20).mean()
                if current_volume > avg_volume * 1.5:
                    analysis['Volume'] = 'above average'
                elif current_volume < avg_volume * 0.5:
                    analysis['Volume'] = 'below average'
                else:
                    analysis['Volume'] = 'normal'
            
            # Price momentum
            if len(df) >= 5:
                recent_prices = df['Close'].tail(5)
                if recent_prices.iloc[-1] > recent_prices.iloc[0]:
                    analysis['Momentum'] = 'positive'
                else:
                    analysis['Momentum'] = 'negative'
                    
        except Exception as e:
            print(f"Error analyzing indicators: {str(e)}")
        
        return analysis
    
    def _format_indicators_text(self, indicators_analysis):
        """Format indicators analysis into readable text"""
        if not indicators_analysis:
            return "limited technical data"
        
        parts = []
        
        if 'RSI' in indicators_analysis:
            rsi_status = indicators_analysis['RSI']
            if rsi_status == 'overbought':
                parts.append("RSI showing overbought conditions")
            elif rsi_status == 'oversold':
                parts.append("RSI indicating oversold levels")
            else:
                parts.append(f"RSI in {rsi_status} territory")
        
        if 'MACD' in indicators_analysis:
            parts.append(f"MACD showing {indicators_analysis['MACD']}")
        
        if 'BB' in indicators_analysis:
            parts.append(f"price {indicators_analysis['BB']} of Bollinger Bands")
        
        if 'Volume' in indicators_analysis:
            parts.append(f"trading volume {indicators_analysis['Volume']}")
        
        if 'Momentum' in indicators_analysis:
            parts.append(f"{indicators_analysis['Momentum']} price momentum")
        
        if not parts:
            return "mixed technical signals"
        
        # Join parts naturally
        if len(parts) == 1:
            return parts[0]
        elif len(parts) == 2:
            return f"{parts[0]} and {parts[1]}"
        else:
            return f"{', '.join(parts[:-1])}, and {parts[-1]}"
    
    def _get_confidence_text(self, confidence):
        """Generate confidence-related text"""
        if confidence >= 0.8:
            return "The AI model shows high confidence in this prediction."
        elif confidence >= 0.6:
            return "The prediction comes with moderate confidence."
        else:
            return "The confidence level is lower, suggesting caution is advised."
    
    def _format_patterns_text(self, patterns):
        """Format detected patterns into readable text"""
        if not patterns:
            return "no significant patterns were detected"
        
        if len(patterns) == 1:
            return f"a {patterns[0].lower()} was detected"
        else:
            return f"patterns including {', '.join(p.lower() for p in patterns)} were identified"
    
    def _get_action_text(self, action, prediction):
        """Generate action recommendation text"""
        action_phrases = {
            'BUY': [
                "Consider this as a potential buying opportunity.",
                "This could be a good entry point for long positions.",
                "The signals suggest a favorable time to accumulate shares."
            ],
            'SELL': [
                "Consider taking profits or reducing position size.",
                "This might be a good time to exit long positions.",
                "The indicators suggest caution and possible selling pressure."
            ],
            'HOLD': [
                "The current recommendation is to hold existing positions.",
                "It may be best to wait for clearer signals before acting.",
                "Consider maintaining current position until trends clarify."
            ]
        }
        
        phrases = action_phrases.get(action, action_phrases['HOLD'])
        return random.choice(phrases)
    
    def _get_educational_text(self, prediction, indicators_analysis):
        """Generate educational content based on the analysis"""
        educational_notes = {
            'UP': [
                "Bullish signals often indicate institutional buying or positive market sentiment. However, always consider overall market conditions.",
                "When multiple indicators align bullishly, it increases the probability of upward movement, but never guarantee it.",
                "Remember that technical analysis is probabilistic - even strong signals can fail in volatile markets."
            ],
            'DOWN': [
                "Bearish signals may indicate profit-taking, negative news, or broader market weakness. Context is crucial.",
                "Multiple bearish indicators increase the likelihood of decline, but markets can be unpredictable.",
                "Consider your risk tolerance and position sizing when acting on bearish signals."
            ],
            'SUSTAIN': [
                "Sideways movement often occurs during consolidation phases before major moves.",
                "Mixed signals suggest market indecision - patience is often the best strategy.",
                "Use consolidation periods to plan your strategy for the eventual breakout."
            ]
        }
        
        notes = educational_notes.get(prediction, educational_notes['SUSTAIN'])
        base_note = random.choice(notes)
        
        # Add specific indicator education
        if 'RSI' in indicators_analysis:
            rsi_status = indicators_analysis['RSI']
            if rsi_status in ['overbought', 'oversold']:
                base_note += f" The RSI being {rsi_status} means the stock may be due for a reversal, but strong trends can remain {rsi_status} for extended periods."
        
        return base_note
