from smolagents import CodeAgent, DuckDuckGoSearchTool, LiteLLMModel, VisitWebpageTool, Tool, tool
import os

@tool
def suggest_menu_prompt(occasion: str) -> str:
    """
    Generates a creative prompt for the LLM to design a menu.

    Args:
        occasion: The type of party or celebration for which the menu should be designed (e.g. "birthday", "holiday brunch").

    Returns:
        A fully-formatted prompt string that can be provided to the LLM.
    """
    return f"""
    Design a creative {occasion} party menu with 5 unique items. 
    Include: 
    - 2 savory dishes with creative names 
    - 2 drinks with thematic ingredients 
    - 1 dessert with a surprise element
    Format as: [Dish Name]: [Description] (max 15 words per item)
    """

@tool
def catering_prompt(query: str) -> str:
    """
    Generates a prompt instructing the LLM to invent imaginative catering services for a specific event.

    Args:
        query: Short description of the customer request or party context (e.g. "vegetarian superhero birthday").

    Returns:
        A fully-formatted prompt string that can be passed to the LLM.
    """
    return f"""
    Find 3 imaginative catering services in Gotham for {query}. 
    For each:
    - Invent a unique specialty dish
    - Create a fictional rating (4.5-5.0)
    - Describe their service style in 10 words
    Format as: [Name] | [Specialty] | Rating: [X.X] | [Style]
    """

class SuperheroThemePromptTool(Tool):
    name = "superhero_theme_prompt_generator"
    description = "Generates prompts for creative superhero party themes."

    inputs = {
        "category": {
            "type": "string",
            "description": "Party theme category (e.g. 'villain masquerade')",
        }
    }
    output_type = "string"

    def forward(self, category: str) -> str:
        return f"""
        Design a {category} superhero party with:
        1. A creative theme name
        2. 3 interactive activities (e.g. "Riddler's puzzle hunt")
        3. 2 signature cocktails with fun names
        4. Decor concept in 10 words
        Format as: [Theme Name]\nActivities: [list]\nCocktails: [list]\nDecor: [description]
        """

agent = CodeAgent(
    tools=[
        DuckDuckGoSearchTool(),
        VisitWebpageTool(),
        suggest_menu_prompt,
        catering_prompt,
        SuperheroThemePromptTool()
    ],
    model=LiteLLMModel(
        model_id="gpt-4o-mini",
        max_tokens=1000,  
        temperature=0.6,
        custom_role_conversions=None,
        api_key=os.environ["OPENAI_TOKEN"],
    ),
    max_steps=10,
    verbosity_level=2
)

agent.run(
    "Gotham's villains have hacked the party playlist! Rewrite the 10-song set so each track secretly encodes "
    "a moral lesson for the guests. For every song: give the track, the hidden lesson, and a one-sentence segue "
    "Alfred can use when it plays."
)

agent.run(
    "Commission a 3-minute elevator pitch to convince Bruce Wayne that a time-travel-themed charity gala "
    "will outshine last year's masquerade. Include: (a) a catchy event title, "
    "(b) three interactive attractions that merge past & future tech, "
    "(c) a gourmet appetiser menu inspired by each attraction, and "
    "(d) one killer closing line Alfred can use to seal the deal."
)
