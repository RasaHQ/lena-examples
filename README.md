# Pass auth header from web widget > rasa pro > action server

This example demonstrates how to pass auth tokens in the headers from a frontend client through Rasa Pro to the action server.

## How it works

1. Frontend sends request with auth header to custom REST channel
2. Custom channel (`auth_rest_channel.py`) forwards headers to action server
3. Action server logs and responds with the received headers

## Configuration

The `credentials.yml` file registers the custom REST channel:

```yaml
custom_components.auth_rest_channel.AuthRestChannel:
```

This tells Rasa to:

- Load the `AuthRestChannel` class from `custom_components/auth_rest_channel.py`
- Make it available at the endpoint `/webhooks/auth_rest_channel/webhook`
- Use this channel instead of the default REST channel for header forwarding

## Testing

1. Start Rasa Pro:

   ```bash
   rasa run --enable-api --cors "*"
   ```

2. Start the action server:

   ```bash
   rasa run actions
   ```

3. Send a POST request to test header passing:

   ```bash
   curl -X POST {{baseUrl}}/webhooks/auth_rest_channel/webhook \
         -H "x-auth-token: very_secure_token" \
         -H "test-header: hello-world" \
     -H "Content-Type: application/json" \
     -d '{"sender": "header-test", "message": "/greet_user"}'
   ```

The bot will respond with the headers it received, demonstrating successful header propagation from frontend → Rasa Pro → Action Server.

## Sample Response

When you send a request with custom headers, you'll receive a response like this:

```json
[
    {
        "recipient_id": "header-test",
        "text": "Hello! How can I assist you?"
    },
    {
        "recipient_id": "header-test",
        "text": "Headers used for API call:\nx-auth-token: very_secure_token\ntest-headerc: hello-world\ncontent-type: application/json\nuser-agent: PostmanRuntime/7.48.0\naccept: */*\ncache-control: no-cache\npostman-token: 32f56f22-d599-47fe-865b-9f6f35e27b17\nhost: localhost:5005\naccept-encoding: gzip, deflate, br\nconnection: keep-alive\ncontent-length: 58"
    },
    {
        "recipient_id": "header-test",
        "text": "What else can I help you with?"
    }
]
```

This shows that your custom headers (like `x-auth-token` and `lena-test`) successfully passed through the entire chain.
