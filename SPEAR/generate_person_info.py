import json
import time
from openai import OpenAI
from typing import List, Dict
import os
import requests

# Google Search API configuration
API_KEY = "your google_api"
SEARCH_ENGINE_ID = "your google SEARCH_ENGINE_ID"

def google_search(query: str, num_results: int = 5) -> str:
    """使用Google Custom Search API搜索信息"""
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "q": query,
        "key": API_KEY,
        "cx": SEARCH_ENGINE_ID,
        "num": num_results
    }
    try:
        resp = requests.get(url, params=params)
        data = resp.json()

        results = []
        if "items" in data:
            for item in data["items"]:
                title = item.get("title", "")
                snippet = item.get("snippet", "")
                link = item.get("link", "")
                results.append(f"【标题】{title}\n【摘要】{snippet}\n【链接】{link}")
        
        return "\n\n".join(results) if results else "未找到相关信息"
    except Exception as e:
        print(f"Search error: {e}")
        return "搜索过程中出现错误"

def load_config(config_path="config.json"):
    """加载配置文件"""
    with open(config_path, "r") as config_file:
        return json.load(config_file)

def create_client():
    """创建 OpenAI 客户端实例"""
    config = load_config()
    return OpenAI(api_key=config["api_key"], base_url=config["api_base_url"].rstrip('/'))

def generate_person_info(client: OpenAI, company_name: str, person_name: str) -> Dict:
    """使用大模型生成个人信息，结合网络搜索结果"""
    # 首先进行网络搜索
    search_query = f"{company_name} {person_name}"
    search_results = google_search(search_query)
    
    prompt = f"""Please generate a detailed personal information template based on the following input:

University - School/College Name: {company_name}  
Person Name: {person_name}  

Web Search Results:  
{search_results}

Using the information above—especially the web search results—generate a JSON-formatted personal profile with the following fields:

- name: Full name  
- email: A valid email address using the institution's domain  
- company: The university or organization name  
- position: Accurate professional title (preferably taken directly from search results)  
- department: Relevant department or division (preferably from search results)  
- recent_activities: 3–4 recent professional activities (preferably from search results)  
- interests: 2–3 interest areas relevant to the person's role  
- location: A reasonable city and region based on the affiliation

Requirements:
1. Use realistic and professional information, prioritizing actual data found in the search results.  
2. Ensure the position and department align with the nature of the institution.  
3. Activities should be concrete and verifiable; use real events from the search results if available.  
4. Interests must be relevant to the role.  
5. Email must match the institution's domain format.  
6. Output only the JSON object—do not include any explanations or extra text.  
7. If no related information is found, generate plausible and context-appropriate default values.

Example output:
{{
    "name": "Full Name",
    "email": "name@university.edu",
    "company": "University Name",
    "position": "Title",
    "department": "Department",
    "recent_activities": [
        "Activity 1",
        "Activity 2",
        "Activity 3"
    ],
    "interests": ["Interest 1", "Interest 2"],
    "location": "City, State/Province"
}}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1000
        )
        
        # 从响应中提取JSON
        content = response.choices[0].message.content
        # 找到JSON字符串的开始和结束
        start = content.find('{')
        end = content.rfind('}') + 1
        if start >= 0 and end > start:
            json_str = content[start:end]
            generated_info = json.loads(json_str)
            
            # 返回生成的信息和搜索数据（分开）
            return {
                "generated_info": generated_info,
                "search_data": {
                    "query": search_query,
                    "results": search_results,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
            }
        else:
            raise ValueError("No valid JSON found in response")
            
    except Exception as e:
        print(f"Error generating info for {person_name} at {company_name}: {e}")
        # 即使生成失败，也保存搜索结果
        return {
            "generated_info": None,
            "search_data": {
                "query": search_query,
                "results": search_results,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "error": str(e)
            }
        }

def process_person_list(client: OpenAI, person_list: List[str]) -> tuple[List[Dict], List[Dict]]:
    """处理人员列表并生成信息，返回模板列表和搜索数据列表"""
    templates = []
    search_results = []
    
    for item in person_list:
        parts = item.split('+')
        if len(parts) != 2:
            print(f"Invalid format for item: {item}")
            continue
            
        company_name = parts[0].strip()
        person_name = parts[1].strip()
        
        print(f"Generating info for {person_name} at {company_name}...")
        print("Searching for online information...")
        person_data = generate_person_info(client, company_name, person_name)
        
        if person_data:
            # 添加搜索数据
            search_results.append(person_data["search_data"])
            
            # 检查是否成功生成了个人信息
            if person_data.get("generated_info"):
                templates.append(person_data["generated_info"])
                print(f"✓ Successfully generated info for {person_name}")
            else:
                print(f"⚠ Search completed but info generation failed for {person_name}")
            # 添加延时以避免API限制
            time.sleep(3)  # 增加延时以避免Google API限制
        else:
            print(f"✗ Failed to process {person_name}")
    
    return templates, search_results

def save_results(templates: List[Dict], search_results: List[Dict], output_file: str = "generated_person_info.json"):
    """保存生成的结果到文件"""
    # 统计成功和失败的数量
    total_count = len(search_results)
    success_count = len(templates)
    failed_count = total_count - success_count
    
    # 创建包含模板和搜索结果的数据结构
    output_data = {
        "templates": templates,
        "google search": search_results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)
    
    print(f"\nResults saved to {output_file}")
    print(f"Total processed: {total_count}")
    print(f"Successfully generated templates: {success_count}")
    print(f"Failed generation: {failed_count}")
    print(f"Success rate: {(success_count/total_count*100):.1f}%" if total_count > 0 else "Success rate: 0%")

def load_person_list(file_path: str = "person_list.txt") -> List[str]:
    """从文件中加载人员列表"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # 读取所有行并去除空白字符
            person_list = [line.strip() for line in f.readlines() if line.strip()]
        print(f"Loaded {len(person_list)} persons from {file_path}")
        return person_list
    except FileNotFoundError:
        print(f"File {file_path} not found. Please create the file with person data.")
        return []
    except Exception as e:
        print(f"Error loading person list from {file_path}: {e}")
        return []

def main():
    # 从文件中加载人员列表
    person_list = load_person_list()
    
    if not person_list:
        print("No person data loaded. Please check the person_list.txt file.")
        return

    # 创建客户端
    client = create_client()
    
    # 处理人员列表
    templates, search_results = process_person_list(client, person_list)
    
    # 保存结果（即使templates为空，也保存搜索结果）
    if templates or search_results:
        save_results(templates, search_results)
    else:
        print("No results were generated.")

if __name__ == "__main__":
    main() 
    
