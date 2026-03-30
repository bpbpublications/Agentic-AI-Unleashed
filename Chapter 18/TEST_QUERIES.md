# TEST_QUERIES.md
# ============================================================
# Suggested Test Queries — Clinical Research Assistant Agent
# Agentic AI Unleashed — Appendix Demo
#
# This file provides a curated set of test queries organized
# by the concept they best exercise. Use these to explore the
# agent's behavior, verify routing decisions, and observe
# LangSmith traces.
# ============================================================


## 🔵 Category 1: Clinical Guidelines (→ Guidelines Agent + RAG)

These queries should route to the Guidelines Agent, trigger a
`search_guidelines` MCP tool call, and return a ClinicalResponse
with evidence sources and confidence level.

1. "What are the USPSTF recommendations for hypertension screening
   in adults under 40?"

2. "At what blood pressure threshold should I start antihypertensive
   medication according to ACC/AHA guidelines?"

3. "What are the screening recommendations for type 2 diabetes?
   Who should be screened and how often?"

4. "What HbA1c target should most type 2 diabetic patients aim for?"

5. "Should a 65-year-old patient with no prior CVD take aspirin
   for primary prevention?"

6. "When is statin therapy recommended for primary cardiovascular
   prevention? What CVD risk threshold triggers a Grade B recommendation?"

7. "What colorectal cancer screening options are recommended for a
   50-year-old patient, and how frequently?"

8. "A 38-year-old patient has obesity and a family history of diabetes.
   Should they be screened for prediabetes?"


## 🟢 Category 2: Drug Information (→ Drug Agent + MCP tool)

These queries should route to the Drug Information Agent, trigger a
`get_drug_info` MCP tool call, and return a DrugInfoResponse with
indication, interactions, and safety note.

9.  "What are the contraindications for metformin? Are there kidney
    function thresholds I should know about?"

10. "What are the main drug interactions I should watch for with
    lisinopril?"

11. "Can atorvastatin cause muscle damage? What are the risk factors
    for statin-induced myopathy?"

12. "Is lisinopril safe in pregnancy?"

13. "What should I tell a patient starting metformin about side
    effects and how to minimize them?"

14. "A patient on atorvastatin is being prescribed erythromycin.
    Is there a drug interaction concern?"


## 🟡 Category 3: Memory & Follow-Up (tests conversational persistence)

Run these as sequential turns in the same session to verify the
agent maintains context across turns.

Turn A: "What are the first-line treatment options for hypertension?"
Turn B: "Of the options you just listed, which ones are contraindicated
         in pregnancy?"
Turn C: "You mentioned ACE inhibitors — is lisinopril one of those,
         and what are its specific warnings?"

---

Turn A: "What is the recommended HbA1c target for type 2 diabetes?"
Turn B: "When would a more lenient target be appropriate?"
Turn C: "What drug would typically be started first, and what do I
         need to check before prescribing it?"


## 🔴 Category 4: HITL Behavior (observe the interrupt/resume cycle)

These queries are high-stakes and should trigger the Human-in-the-Loop
checkpoint (requires_review=True). In interactive mode (python main.py
--interactive), you will be prompted to approve or modify the response.

15. "My patient has stage 3 CKD and type 2 diabetes — should they
    stay on metformin?"

16. "What is the recommended treatment approach for a hypertensive
    patient who also has heart failure?"

17. "A diabetic patient with established cardiovascular disease —
    should their first-line agent be metformin or a GLP-1 agonist?"


## ⚪ Category 5: Out-of-Scope (→ Direct Response, no tool calls)

These should route to direct_response without any MCP tool invocations.

18. "Hello, what can you help me with?"

19. "What is LangGraph and how does this demo use it?"

20. "What is the weather like in Seattle today?"

21. "Can you recommend a restaurant for lunch?"


## 🔬 Bonus: Stress Tests (edge cases and boundary behavior)

22. "What is the recommended statin dose for primary prevention?"
    (Tests: the corpus has guideline info but not dosing — observe
    how the agent handles partial evidence)

23. "Tell me about semaglutide for diabetes."
    (Tests: drug not in corpus — observe graceful handling of
    low/no retrieval results)

24. "What does USPSTF stand for and why do their grades matter?"
    (Tests: borderline — could be direct or guidelines; observe
    supervisor routing decision)


## 📊 How to Observe in LangSmith

When LANGCHAIN_TRACING_V2=true, each query produces a trace showing:
  • Supervisor node: routing decision + SupervisorDecision JSON
  • Specialist node: tool call request → MCP response → structured output
  • HITL node: interrupt value and resume input
  • Format node: final assembly
  • Token counts and latency for every node

Navigate to https://smith.langchain.com → your project →
click any trace to explore the full execution tree.
