# Example Queries for RAG Agent

This document contains example queries you can use to test the RAG Agent with the mock data.

## 📋 Table of Contents
1. [Project-related Questions](#project-related-questions)
2. [Requirement-related Questions](#requirement-related-questions)
3. [User Story-related Questions](#user-story-related-questions)
4. [Relationship Questions](#relationship-questions)
5. [Cross-domain Questions](#cross-domain-questions)
6. [Technical Questions](#technical-questions)

---

## 🎯 Project-related Questions

### Basic Questions
- `What projects do we have?`
- `Tell me about all active projects`
- `List all projects and their status`
- `What is Project Alpha about?`
- `Describe Project Beta`
- `What are the stakeholders for Project Gamma?`

### Detailed Questions
- `What features does Project Alpha include?`
- `What is the current version of Project Beta?`
- `Which projects are currently in progress?`
- `What are the differences between Project Alpha and Project Beta?`
- `Who are the stakeholders for Project Epsilon?`

---

## 📝 Requirement-related Questions

### Authentication & Security
- `What are the requirements for user authentication?`
- `Tell me about the authentication requirements`
- `What is REQ-001 about?`
- `What acceptance criteria does the authentication requirement have?`
- `How does multi-factor authentication work?`

### Access Control
- `What is role-based access control?`
- `How does RBAC work in the system?`
- `What roles are supported?`
- `Tell me about REQ-002`

### API & Integration
- `What are the API requirements?`
- `Tell me about RESTful API endpoints`
- `What is the rate limiting requirement?`
- `How does API Gateway work?`
- `What is REQ-004 about?`

### Database & Performance
- `What are the database optimization requirements?`
- `Tell me about REQ-005`
- `How is database performance optimized?`
- `What caching mechanisms are used?`

### Mobile Features
- `What are the push notification requirements?`
- `How does offline mode work?`
- `Tell me about biometric authentication requirements`
- `What is REQ-006?`
- `What is REQ-007 about?`
- `Describe REQ-010`

### Analytics & Reporting
- `What are the analytics dashboard requirements?`
- `Tell me about REQ-008`
- `What visualization features are required?`

---

## 👤 User Story-related Questions

### Customer/User Stories
- `What user stories are related to order history?`
- `Tell me about US-001`
- `How can users view their order history?`
- `What user stories involve login?`
- `Describe the login user story`

### Admin Stories
- `What user stories are for admins?`
- `Tell me about user role management stories`
- `What is US-003 about?`

### Developer Stories
- `What stories are for developers?`
- `Tell me about API documentation stories`
- `What is US-004?`

### Mobile User Stories
- `What mobile user stories do we have?`
- `Tell me about push notification stories`
- `How does offline mode work for users?`
- `What is the biometric login story?`
- `Describe US-005, US-006, and US-010`

### Analytics Stories
- `What stories are related to analytics?`
- `Tell me about dashboard creation stories`
- `What is US-007 about?`
- `How can users export reports?`

### DevOps Stories
- `What stories are for DevOps engineers?`
- `Tell me about API Gateway monitoring`
- `What is US-009?`
- `How do I monitor system performance?`

---

## 🔗 Relationship Questions

### Project Dependencies
- `What requirements does Project Alpha have?`
- `What user stories belong to Project Beta?`
- `Which requirements are part of Project Gamma?`
- `What is included in Project Epsilon?`
- `Show me everything related to Project Delta`

### Requirement Dependencies
- `What requirements depend on authentication?`
- `What does RBAC depend on?`
- `Show me the dependency chain for REQ-009`
- `What requirements are related to REQ-001?`
- `What depends on API endpoints?`

### User Story to Requirement
- `What requirements are derived from user stories?`
- `Which requirement is derived from US-002?`
- `Show me requirements related to US-001`
- `What requirements come from the login user story?`

### User Story Relationships
- `What user stories relate to each other?`
- `What blocks US-003?`
- `Show me related stories to order history`
- `What stories are related to login?`

---

## 🔄 Cross-domain Questions

### Project & Requirements
- `What are all the requirements for Project Alpha?`
- `Which projects have authentication requirements?`
- `What requirements are shared across projects?`
- `Show me all requirements and which projects they belong to`

### Project & User Stories
- `What user stories are in Project Alpha?`
- `Which projects have mobile user stories?`
- `Show me all user stories organized by project`
- `What stories are planned for Project Gamma?`

### Requirements & User Stories
- `What user stories implement the authentication requirement?`
- `Show me all user stories and their related requirements`
- `Which stories help fulfill REQ-004?`
- `What requirements are covered by user stories?`

### Complete Views
- `Give me a complete overview of Project Alpha including requirements and user stories`
- `Show me the full picture of authentication: requirements, user stories, and related projects`
- `What is the complete status of mobile features across all projects?`
- `Provide a comprehensive view of API-related features`

---

## 🔧 Technical Questions

### Architecture & Design
- `What is the system architecture?`
- `How is the system designed for scalability?`
- `What microservices are mentioned?`
- `Tell me about the API Gateway architecture`

### Performance & Optimization
- `How is database performance optimized?`
- `What caching strategies are used?`
- `How does the system handle load balancing?`
- `What are the performance requirements?`

### Security
- `What security measures are implemented?`
- `How does authentication and authorization work?`
- `What are the security requirements?`
- `Tell me about MFA implementation`

### Integration
- `How can external systems integrate?`
- `What API endpoints are available?`
- `What documentation is provided for developers?`
- `How does the API Gateway handle requests?`

### Mobile
- `How does offline mode work?`
- `What mobile platforms are supported?`
- `How are push notifications delivered?`
- `What is the mobile app architecture?`

### Analytics
- `What analytics capabilities exist?`
- `How can users create custom dashboards?`
- `What visualization options are available?`
- `How can reports be exported?`

---

## 💡 Advanced Query Examples

### Multi-faceted Questions
- `What are all the high-priority requirements across all projects?`
- `Show me all critical user stories and their status`
- `What requirements are approved but not yet implemented?`
- `Which user stories are in progress and what requirements do they relate to?`
- `Give me a summary of all authentication-related features`

### Status & Progress
- `What is the status of all requirements?`
- `Which user stories are done?`
- `What stories are blocked?`
- `Show me all in-progress work across projects`
- `What is the backlog status?`

### Priority-based
- `What are all critical requirements?`
- `Show me high-priority user stories`
- `What critical items need immediate attention?`
- `List all medium-priority items`

### Sprint-based
- `What stories are in Sprint 1?`
- `Show me all Sprint 2 items`
- `What is planned for Sprint 3?`
- `What is the story point distribution across sprints?`

---

## 🎨 Query Tips

### Best Practices
1. **Be specific**: Instead of "tell me about projects", ask "what projects do we have?"
2. **Use IDs when known**: "What is REQ-001?" or "Tell me about US-002"
3. **Ask about relationships**: "What depends on authentication?" or "What stories relate to login?"
4. **Combine concepts**: "Show me all authentication-related items" or "What mobile features do we have?"

### Example Good Queries
✅ `What are the requirements for Project Alpha?`
✅ `Tell me about all authentication-related features`
✅ `What user stories implement REQ-001?`
✅ `Show me the complete picture of mobile features`

### Example Vague Queries (may work but less specific)
⚠️ `Tell me about the system`
⚠️ `What do we have?`
⚠️ `Show me everything`

---

## 📊 Statistics Questions (may require custom queries)

- `How many projects do we have?`
- `How many requirements are there?`
- `What is the total story points?`
- `How many user stories are in each sprint?`
- `What is the distribution of requirement types?`
- `How many relationships exist between requirements?`

---

## 🚀 Quick Start Test Queries

Try these simple queries to get started:

1. `What projects do we have?`
2. `Tell me about Project Alpha`
3. `What are the authentication requirements?`
4. `What user stories are related to login?`
5. `What requirements does Project Alpha have?`
6. `Show me all high-priority requirements`
7. `What mobile features do we have?`
8. `Tell me about API requirements`

---

**Note**: The RAG Agent searches both Milvus (KB - Knowledge Base) for document chunks and Neo4j (KG - Knowledge Graph) for structured data. Some queries may return results from both sources, providing comprehensive answers with references.

