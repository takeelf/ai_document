from bs4 import BeautifulSoup
import requests


class Website:
    url: str
    title: str
    text: str
    
    def __init__(self, url):
        self.url = url
        response = requests.get(url)
        self.body = response.content
        soup = BeautifulSoup(self.body, 'html.parser')
        self.title = soup.title.string if soup.title else "No title"
        for irrelevant in soup(["script", "style", "img", "input"]):
            irrelevant.decompose()
        self.text = soup.body.get_text(separator="\n", strip=True)
        
    def get_contents(self):
        return f"Webpage Title:\n{self.title}\n\nWebpage Contents:\n {self.text}\n"