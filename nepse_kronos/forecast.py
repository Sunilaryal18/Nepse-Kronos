import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import Kronos, KronosPredictor, KronosTokenizer
from nepse_kronos.trading_calendar import next_trading_days

FEATURES = ["open", "high", "low", "close", "volume", "amount"]


def load_predictor(model_name="NeoQuasar/Kronos-small", tokenizer_name="NeoQuasar/Kronos-Tokenizer-base",
                   device=None, max_context=512):
    """Load Kronos from the Hugging Face Hub or a local checkpoint directory."""
    # from_pretrained leaves the modules in training mode; eval() turns dropout off for inference.
    tokenizer = KronosTokenizer.from_pretrained(tokenizer_name).eval()
    model = Kronos.from_pretrained(model_name).eval()
    return KronosPredictor(model, tokenizer, device=device, max_context=max_context)


def forecast_next(predictor, df, pred_len, lookback=400, holidays=(), sample_count=10, T=1.0, top_p=0.9):
    """Forecast the pred_len NEPSE trading days after the last row of df."""
    if len(df) < lookback:
        raise ValueError(f"Need at least {lookback} rows of history, got {len(df)}")
    history = df.iloc[-lookback:].reset_index(drop=True)
    future = next_trading_days(history["timestamps"].iloc[-1], pred_len, holidays)
    pred = predictor.predict(
        df=history[FEATURES],
        x_timestamp=history["timestamps"],
        y_timestamp=future,
        pred_len=pred_len,
        T=T,
        top_p=top_p,
        sample_count=sample_count,
        verbose=False,
    )
    pred.index.name = "timestamps"
    return pred


def plot_forecast(history, forecast, path, title):
    """Last 120 days of close/volume plus the forecast."""
    recent = history.tail(120)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax1.plot(recent["timestamps"], recent["close"], label="History", color="tab:blue")
    ax1.plot(forecast.index, forecast["close"], label="Forecast", color="tab:red")
    ax1.set_ylabel("Close (NPR)")
    ax1.set_title(title)
    ax1.legend(loc="upper left")
    ax1.grid(True)
    ax2.bar(recent["timestamps"], recent["volume"], color="tab:blue")
    ax2.bar(forecast.index, forecast["volume"], color="tab:red")
    ax2.set_ylabel("Volume")
    ax2.grid(True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
