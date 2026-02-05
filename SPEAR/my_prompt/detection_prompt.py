detection_prompt = """
I want you to act as a phishing email detection expert. Based on the provided email content, determine whether it is a phishing email or a legitimate one.

Phishing emails often impersonate legitimate organizations and companies, using social engineering techniques to deceive users. These techniques include but are not limited to: fake rewards, fake accounts, security warnings, creating urgency, personalized targeting, critical notifications, malicious attachments, spoofed sender addresses or domains, embedded malicious URLs, harmful file attachments, and file types such as .lnk or .hat.

Follow these steps to evaluate the email:

1. Identify if the email impersonates a legitimate entity or company. Check for issues in the email format, such as generic greetings, sensitive information, or unusual sender details.  
2. Examine whether the email contains common phishing characteristics (e.g., urgency, promised rewards, critical notifications). Pay attention to whether the recipient address has been replaced with a fake or masked address.  
3. Look for social engineering tactics that entice the user to click on a hyperlink. Inspect the URL to determine whether it is misleading, directing to suspicious websites, or mimicking legitimate domains.  
4. Analyze whether the email contains excessive advertisements or promotional content, is frequently sent without permission, or has repetitive promotional messages.

Based on the above criteria, determine whether the email is a phishing attempt and provide a brief analysis.  
- If the email is a phishing attempt, return **'Determined as a phishing email'**.  
- If the email is legitimate, return **'Determined as another legitimate email'**.

Email Content:  
{email_content}
"""

# detection_prompt = """Reasoning requirement: 1.  Evaluate if the agent interacts with others in a natural and realistic manner (here are a few common questions to check: a. whether the agent is confusing with its own identity?  b. whether the agent repeats others’ words/actions without any reason?  c. whether the agent is being overly polite considering the context?).  Start the analysis with tag <naturalness> 2.  Analyze whether the actions of the agent align with their character traits (e.g., personality, values, and etc.).  Start the analysis with tag <consistency>.  Output your reasoning process to the ‘reasoning’ field.  Output an integer score ranging from 0 and 10 in the ’score’ field.  A higher score indicates that the agent is more believable. 

# email content:
# {email_content}"""