import smtplib
from email.mime.text import MIMEText

def send_email_alert(prediction: str, confidence: float, to_email: str,
                     from_email: str, app_password: str):
    subject = "📈 Crypto AI Prediction Alert"
    body = f"The model predicts: {prediction.upper()} with confidence {confidence:.2f}%"
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = from_email
    msg['To'] = to_email

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(from_email, app_password)
            server.send_message(msg)
        return True
    except Exception as e:
        return f"Email failed: {e}"
