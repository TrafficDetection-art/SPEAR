web_generation_prompt = """
You are an AI assistant designed to analyze email content and generate phishing-related web resources.
Based on the email body and the URL, you need to help me generate the corresponding website resource.

email content：{email_content}
url：{url}
""" 