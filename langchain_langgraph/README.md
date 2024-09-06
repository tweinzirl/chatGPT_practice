### Lesson 1: Simple ReAct Agent from Scratch
 - Lesson_1_Student.ipynb
 - ReAct = Reasoning and Acting
 - thought-action-observation loop facilitated by the prompt
```
You run in a loop of Thought, Action, PAUSE, Observation.
At the end of the loop you output an Answer
Use Thought to describe your thoughts about the question you have been asked.
Use Action to run one of the actions available to you - then return PAUSE.
Observation will be the result of running those actions.

Your available actions are:

calculate:
e.g. calculate: 4 * 7 / 3
Runs a calculation and returns the number - uses Python so be sure to use floating point syntax if necessary

average_dog_weight:
e.g. average_dog_weight: Collie
returns average weight of a dog when given the breed

Example session:

Question: How much does a Bulldog weigh?
Thought: I should look the dogs weight using average_dog_weight
Action: average_dog_weight: Bulldog
PAUSE

You will be called again with this:

Observation: A Bulldog weights 51 lbs

You then output:

Answer: A bulldog weights 51 lbs
```
 - Agent class stores the messages as a list and provides an execute function for calling the LLM.
 - The prompt instructions an action to be determined based on the input. Then the LLM generates a PAUSE and waits for an observation to be added concerning the output of the corresponding tool. When the LLM decides no more actions are needed, the loop finishes.

### Lesson 2: LangGraph Components
 - Lesson_2_Student.ipynb
 - LangGraph is an extension of Langchain that supports graphs; single and multi-agent flows are described as graphs
   - allows for extremely controlled workflows
   - allows human-in-the-loop workflows
 - graph components:
   - nodes = agents or functions
   - edges = connect nodes
   - conditional edges = decisions
 - Agent State is to all parts of the graph at all times

```python
class Agent:

    def __init__(self, model, tools, system=""):
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_openai)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges(
            "llm",
            self.exists_action,
            {True: "action", False: END}
        )
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        self.graph = graph.compile()
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    def exists_action(self, state: AgentState):
        result = state['messages'][-1]
        return len(result.tool_calls) > 0

    def call_openai(self, state: AgentState):
        messages = state['messages']
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        return {'messages': [message]}

    def take_action(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            print(f"Calling: {t}")
            if not t['name'] in self.tools:      # check for bad tool name from LLM
                print("\n ....bad tool name....")
                result = "bad tool name, retry"  # instruct LLM to retry if bad
            else:
                result = self.tools[t['name']].invoke(t['args'])
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        print("Back to the model!")
        return {'messages': results}
```

### Lesson 3: Agentic Search
 - Example use of Tavily (json output) and DuckDuckGo search (output links that need to be scraped)


### Lesson 4: AI Agents in LangGraph
 - Persistance: Keep around Agent State at different timesteps. Enables time travel and realtime edits.
 - Streaming: emit a list of signals about what is happening at a given instant
 - Can stream events as they happen or tokens as they are produced.
 - Use 'threads' to track different conversations for user sessions.

```python
for event in abot.graph.stream({"messages": messages}, thread):
    for v in event.values():
        print(v['messages'])
```

```python
messages = [HumanMessage(content="What is the weather in SF?")]
thread = {"configurable": {"thread_id": "4"}}
async for event in abot.graph.astream_events({"messages": messages}, thread, version="v1"):
    kind = event["event"]
    if kind == "on_chat_model_stream":
        content = event["data"]["chunk"].content
        if content:
            # Empty content in the context of OpenAI means
            # that the model is asking for a tool to be invoked.
            # So we only print non-empty content
            print(content, end="|")
```

### Lession 5: Human in the loop
 - force human in loop with interrupt_before:
```python
        self.graph = graph.compile(
            checkpointer=checkpointer,
            interrupt_before=["action"]
        )
```
 - Each thread keeps a history of states denoted by thread_ts. For any thread can immediately access current state or any prior state. Graph can be involed/streamed from that state without or without manual correction.

### Lesson 6: Essay Writer
 - More complicated example where essay is by graph with many nodes: plan, research, generate, reflect. Iteration continues until max number of iterations reached.
 - Notebook includes gradio app to view stepwise execution of graph.
 - Note, Tavily API key is needed.
