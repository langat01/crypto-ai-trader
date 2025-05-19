from twilio.rest import Client

def send_sms_alert(prediction: str, confidence: float,
                   to_number: str, from_number: str,
                   twilio_sid: str, twilio_token: str):
    body = f"Crypto AI predicts {prediction.upper()} with {confidence:.2f}% confidence."
    try:
        client = Client(twilio_sid, twilio_token)
        client.messages.create(body=body, from_=from_number, to=to_number)
        return True
    except Exception as e:
        return f"SMS failed: {e}"
