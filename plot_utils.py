import plotly.graph_objs as go

def plot_price(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close'))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], mode='lines', name='SMA 20'))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], mode='lines', name='SMA 50'))
    fig.update_layout(title="BTC Price with SMAs", height=400)
    return fig

def plot_macd(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD'))
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal Line'))
    fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], name='Histogram'))
    fig.update_layout(title="MACD Indicators", height=300)
    return fig
