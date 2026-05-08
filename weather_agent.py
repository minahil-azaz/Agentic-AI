import os
import json
import requests
from dotenv import load_dotenv
from groq import Groq
from typing import Dict, List, Optional
from datetime import datetime

# Load environment variables
load_dotenv()

class WeatherAgent:
    def __init__(self):
        """Initialize the Weather Agent with Groq LLM"""
        self.groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))
        self.weather_api_key = os.getenv('OPENWEATHER_API_KEY')
        self.conversation_history = []
        
        # Updated to use current Groq models
        self.available_models = [
            "llama-3.3-70b-versatile",  # Latest Llama model
            "llama-3.1-8b-instant",     # Fast, efficient model
            "gemma2-9b-it",             # Google's Gemma model
            "llama-guard-3-8b"          # For safety
        ]
        self.current_model = "llama-3.3-70b-versatile"  # Using the latest versatile model
    
    def get_city_temperature_real(self, city_name: str) -> Optional[Dict]:
        """Fetch real temperature data from OpenWeatherMap API"""
        if not self.weather_api_key:
            return None
            
        try:
            url = f"http://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={self.weather_api_key}&units=metric"
            response = requests.get(url)
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "city": city_name,
                    "temperature": data['main']['temp'],
                    "feels_like": data['main']['feels_like'],
                    "humidity": data['main']['humidity'],
                    "condition": data['weather'][0]['description'],
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            else:
                return None
        except Exception as e:
            print(f"Error fetching weather data: {e}")
            return None
    
    def parse_user_query(self, user_input: str) -> List[str]:
        """Use Groq LLM to parse the user query and extract city names"""
        prompt = f"""
        You are a weather information parser. Extract all city names from the following user query.
        Return only a JSON array of city names. If no cities are found, return an empty array.
        
        User query: "{user_input}"
        
        Examples:
        - "What's the temperature in London?" -> ["London"]
        - "Compare temperatures of New York, Tokyo, and Paris" -> ["New York", "Tokyo", "Paris"]
        - "How's the weather in Los Angeles?" -> ["Los Angeles"]
        
        Return only the JSON array, no other text.
        """
        
        try:
            response = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.current_model,  # Updated model
                temperature=0.1,
                max_tokens=200
            )
            
            content = response.choices[0].message.content
            # Clean up the response to ensure it's valid JSON
            content = content.strip()
            if content.startswith('```json'):
                content = content[7:]
            if content.startswith('```'):
                content = content[3:]
            if content.endswith('```'):
                content = content[:-3]
            content = content.strip()
            
            cities = json.loads(content)
            return cities
        except Exception as e:
            print(f"Error parsing query: {e}")
            # Fallback: simple keyword extraction
            words = user_input.lower().split()
            common_cities = ['london', 'new york', 'tokyo', 'paris', 'beijing', 
                           'moscow', 'sydney', 'mumbai', 'dubai', 'singapore']
            detected = [word.title() for word in words if word.lower() in common_cities]
            # Also check for multi-word cities
            for i in range(len(words)-1):
                two_words = f"{words[i]} {words[i+1]}"
                if two_words in common_cities:
                    detected.append(two_words.title())
            return detected
    
    def generate_response(self, user_input: str, weather_data: List[Dict]) -> str:
        """Generate a natural language response using Groq"""
        if not weather_data:
            return "I couldn't find weather information for those cities. Please check the city names and try again."
        
        # Prepare weather data summary
        weather_summary = json.dumps(weather_data, indent=2)
        
        prompt = f"""
        You are a helpful weather assistant. Based on the following weather data, generate a friendly response to the user.
        
        User query: "{user_input}"
        
        Weather data:
        {weather_summary}
        
        Requirements:
        1. Be conversational and friendly
        2. If multiple cities, compare them naturally
        3. Include temperatures with proper units (°C)
        4. Add relevant details like feels-like temperature or conditions
        5. Keep response concise but informative (max 150 words)
        6. Use emojis appropriately (🌡️, ☀️, 🌧️, etc.)
        
        Generate response:
        """
        
        try:
            response = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.current_model,  # Updated model
                temperature=0.7,
                max_tokens=500
            )
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating response: {e}")
            # Fallback response
            return self.generate_simple_response(weather_data)
    
    def generate_simple_response(self, weather_data: List[Dict]) -> str:
        """Simple fallback response generator"""
        if len(weather_data) == 1:
            data = weather_data[0]
            return f"The temperature in {data['city']} is {data['temperature']}°C, feeling like {data['feels_like']}°C with {data['condition']}."
        else:
            response = "Here are the temperatures:\n"
            for data in weather_data:
                response += f"• {data['city']}: {data['temperature']}°C (feels like {data['feels_like']}°C, {data['condition']})\n"
            return response
    
    def process_query(self, user_input: str) -> str:
        """Main processing pipeline"""
        print(f"\n🤖 Agent: Processing query...")
        
        # Step 1: Parse cities from user input using LLM
        cities = self.parse_user_query(user_input)
        
        if not cities:
            return "I couldn't identify any city names in your query. Please specify a city (e.g., 'What's the temperature in London?')"
        
        print(f"📍 Cities detected: {', '.join(cities)}")
        
        # Step 2: Fetch weather data for each city
        weather_data = []
        for city in cities:
            print(f"🌡️ Fetching weather for {city}...")
            
            # Try real API first
            data = self.get_city_temperature_real(city)
            
            # Fallback: Simulated data for demo
            if not data:
                print(f"⚠️ Using simulated data for {city}")
                data = self.get_simulated_weather_data(city)
            
            if data:
                weather_data.append(data)
        
        # Step 3: Generate response using LLM
        response = self.generate_response(user_input, weather_data)
        
        # Add to conversation history
        self.conversation_history.append({
            "user": user_input,
            "assistant": response,
            "timestamp": datetime.now().isoformat()
        })
        
        return response
    
    def get_simulated_weather_data(self, city: str) -> Dict:
        """Generate simulated weather data for demo purposes"""
        import random
        simulated_temps = {
            "lahore":(20,40),
            "kahna":(25,40),
            "london": (5, 15),
            "new york": (-5, 25),
            "tokyo": (5, 30),
            "paris": (0, 20),
            "beijing": (-10, 30),
            "moscow": (-20, 20),
            "sydney": (10, 35),
            "mumbai": (20, 35),
            "dubai": (25, 40),
            "singapore": (25, 32)
            
        }
        
        city_lower = city.lower()
        if city_lower in simulated_temps:
            temp_range = simulated_temps[city_lower]
            temp = random.uniform(temp_range[0], temp_range[1])
        else:
            temp = random.uniform(-10, 35)
        
        conditions = ["☀️ clear sky", "⛅ few clouds", "☁️ scattered clouds", 
                     "🌧️ light rain", "🌧️ moderate rain", "🌫️ overcast", 
                     "❄️ light snow", "⚡ thunderstorm"]
        
        return {
            "city": city,
            "temperature": round(temp, 1),
            "feels_like": round(temp + random.uniform(-3, 3), 1),
            "humidity": random.randint(30, 90),
            "condition": random.choice(conditions),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "note": "Simulated data (OpenWeatherMap API key not configured)"
        }
    
    def test_connection(self) -> bool:
        """Test if Groq API is working with current model"""
        try:
            test_response = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": "Say 'API working'"}],
                model=self.current_model,
                max_tokens=20
            )
            print(f"✅ Groq API connected successfully using model: {self.current_model}")
            return True
        except Exception as e:
            print(f"❌ Groq API test failed: {e}")
            return False


class InteractiveWeatherAssistant:
    def __init__(self):
        self.agent = WeatherAgent()
        
    def run(self):
        """Run interactive command-line interface"""
        print("\n" + "="*60)
        print("🌤️  WEATHER ASSISTANT AGENT SYSTEM 🤖")
        print("="*60)
        
        # Test API connection
        print("\n🔌 Testing API connection...")
        if not self.agent.test_connection():
            print("\n⚠️ API connection failed. The assistant may not work properly.")
            print("Please check your GROQ_API_KEY and internet connection.")
        
        print("\nI can tell you the temperature of different cities!")
        print("Example queries:")
        print("  • What's the temperature in London?")
        print("  • Compare temperatures of New York, Tokyo, and Paris")
        print("  • How's the weather in Sydney?")
        print("  • Is it hotter in Dubai or Singapore?")
        print("\nType 'quit' or 'exit' to end the conversation.")
        print("-"*60)
        
        while True:
            user_input = input("\n💬 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\n👋 Goodbye! Have a great day!")
                break
            
            if not user_input:
                continue
            
            response = self.agent.process_query(user_input)
            print(f"\n🤖 Assistant: {response}")


# Quick test function
def quick_test():
    """Quick test function without interactive mode"""
    print("Running quick test...")
    agent = WeatherAgent()
    
    # Test API connection
    if not agent.test_connection():
        print("Please set your GROQ_API_KEY in .env file")
        return
    
    # Test queries
    test_queries = [
        "What's the temperature in London?",
        "Compare Paris and Tokyo"
    ]
    
    for query in test_queries:
        print(f"\n📝 Query: {query}")
        response = agent.process_query(query)
        print(f"✅ Response: {response}")


if __name__ == "__main__":
    # Check for API key
    if not os.getenv('GROQ_API_KEY'):
        print("\n❌ GROQ_API_KEY not found!")
        key = input("Please enter your Groq API key: ").strip()
        if key:
            with open('.env', 'w') as f:
                f.write(f"GROQ_API_KEY={key}")
            print("✓ API key saved to .env file")
            load_dotenv()
        else:
            print("❌ No API key provided. Exiting...")
            exit()
    
    # Run the assistant
    assistant = InteractiveWeatherAssistant()
    assistant.run()