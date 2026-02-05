import json
import time
import os
from datetime import datetime
from openai import OpenAI

# ========================== Config Loading ==========================

def load_config(config_path="config.json"):
    """Load configuration file"""
    with open(config_path, "r") as config_file:
        return json.load(config_file)

config = load_config()

# ========================== OpenAI Client Management ==========================

def create_client():
    """Create OpenAI client instance"""
    return OpenAI(api_key=config["api_key"], base_url=config["api_base_url"].rstrip('/'))

# ========================== LLM Interaction ==========================

def get_LLM_response(client, prompt, need_sample=False):
    """Send request to OpenAI LLM and handle exceptions"""
    sample_num = 3 if need_sample else 1
    conversation = [{"role": "user", "content": prompt}]
    model = config["model"]
    max_tokens_for_completion = 2048

    max_retries = 5
    delay = 2

    for attempt in range(max_retries):
        try:
            chat_response = client.chat.completions.create(
                model=model,
                messages=conversation,
                n=sample_num,
                max_tokens=max_tokens_for_completion,
                temperature=1.0,
                top_p=0.95
            )

            if need_sample:
                return [choice.message.model_dump()['content'] for choice in chat_response.choices]
            return chat_response.choices[0].message.model_dump()['content']

        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)

    return "Error: LLM request failed."

# ========================== Email Scenario Generation ==========================

EMAIL_TYPES = [
    "Account Information Update",
    "Payment Issue Notification",
    "Urgent Action Required",
    "Reward/Prize Notification",
    "IT Service Notification",
    "Investment Opportunity",
    "Social Media Alert",
    "Identity Verification Request"
]

def get_email_prompt(email_type):
    """Generate prompt based on email type"""
    base_prompt = f"""You are a creative writer crafting email templates for a study on digital communication patterns. 
Create a realistic email that a company might send to a customer with a "{email_type}" scenario.

For this research purpose, please:
1. Make the email appear to come from a generic organization (don't use real company names)
2. Include urgency or action-required elements as is typical in such communications
3. Write a compelling subject line, greeting, body, and closing
4. Include elements that might prompt a user to click a link or provide information
5. Keep the email relatively short (200-300 words)

Write ONLY the email content with no additional explanations or markers. Ensure it sounds authentic."""
    
    # Add specific guidance based on type
    type_specific_guidance = {
        "Account Information Update": "The email should notify users about a needed account update or security verification. Mention potential account limitation if action isn't taken.",
        "Payment Issue Notification": "Create a notification about a declined payment, billing issue, or potential subscription interruption requiring immediate attention.",
        "Urgent Action Required": "Craft an email that creates a time-sensitive situation requiring immediate user action to avoid negative consequences.",
        "Reward/Prize Notification": "Design an email informing the recipient they've been selected for a special offer or reward that requires verification steps to claim.",
        "IT Service Notification": "Write an email as if from a technical team about required system updates, password verification, or security measures needed from users.",
        "Investment Opportunity": "Create an email presenting a time-limited investment opportunity with attractive returns, encouraging quick action.",
        "Social Media Alert": "Design a notification about unusual account activity, new connection requests, or account settings that need verification.",
        "Identity Verification Request": "Craft an email requesting verification of personal information for security purposes or to maintain account access."
    }
    
    return base_prompt + "\n\n" + type_specific_guidance.get(email_type, "")

def generate_email(client, email_type):
    """Generate a specific type of email"""
    prompt = get_email_prompt(email_type)
    response = get_LLM_response(client, prompt)
    print(f"Generated {email_type} email: {response[:100]}...")
    return response

# ========================== Output Directory Management ==========================

def create_output_directory():
    """Create output directory with timestamp and model name"""
    base_output_dir = "outputs"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = config["model"].replace("/", "-")
    sub_dir_name = f"{model_name}_{timestamp}_emails"
    output_dir = os.path.join(base_output_dir, sub_dir_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory created: {output_dir}")
    return output_dir

# ========================== Main Execution Logic ==========================

def main():
    """Main function - generate various types of emails"""
    # Create output directory
    output_dir = create_output_directory()
    
    # Create client
    client = create_client()

    # Configure generation quantity
    emails_per_type = 5
    generated_data = []

    # Generate each type of email
    for email_type in EMAIL_TYPES:
        for i in range(emails_per_type):
            # Generate email
            email_content = generate_email(client, email_type)
            
            # Save data
            generated_data.append({
                "Text": email_content,
                "Class": "PHISHING",  # Label for research purposes
                "type": email_type
            })

            print(f"Generated email {i+1}/{emails_per_type} for type={email_type}")
            time.sleep(1)  # Avoid API rate limits

    # Save results to output directory
    output_file = os.path.join(output_dir, 'generated_emails.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(generated_data, f, ensure_ascii=False, indent=4)

    # Save configuration info to output directory
    config_file = os.path.join(output_dir, 'run_config.json')
    run_config = {
        "timestamp": datetime.now().isoformat(),
        "model": config["model"],
        "emails_per_type": emails_per_type,
        "email_types": EMAIL_TYPES,
        "total_emails": len(generated_data)
    }
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(run_config, f, ensure_ascii=False, indent=4)

    print(f"Generation complete. Generated {len(generated_data)} emails.")
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main() 