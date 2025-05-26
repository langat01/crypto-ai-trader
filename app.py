def execute_trade(prediction, current_price, portfolio, fee=0.0002):
    if prediction == 1 and not portfolio['holding']:
        # Long entry
        portfolio['position'] = (portfolio['capital'] * (1 - fee)) / current_price
        portfolio['capital'] = 0
        portfolio['holding'] = 'long'
        portfolio['entry_price'] = current_price
        portfolio['history'].append(('long', current_price, datetime.now()))
        
    elif prediction == -1 and enable_shorting and not portfolio['holding']:
        # Short entry
        portfolio['position'] = (portfolio['capital'] * (1 - fee)) / current_price
        portfolio['capital'] = 0
        portfolio['holding'] = 'short'
        portfolio['entry_price'] = current_price
        portfolio['history'].append(('short', current_price, datetime.now()))
        
    elif prediction == 0 and portfolio['holding']:
        # Exit position
        multiplier = 1 if portfolio['holding'] == 'long' else -1
        exit_value = portfolio['position'] * current_price * (1 + multiplier * (current_price - portfolio['entry_price'])/portfolio['entry_price'])
        portfolio['capital'] = exit_value * (1 - fee)
        portfolio['position'] = 0
        portfolio['holding'] = False
        # Fix: Proper tuple concatenation
        portfolio['history'][-1] = portfolio['history'][-1] + (current_price, datetime.now())
        
    return portfolio
