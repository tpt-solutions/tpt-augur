pub mod agent {
    use std::collections::HashSet;

    #[derive(Default)]
    pub struct RetryPolicy;

    pub struct AgentInfo {
        pub name: String,
        pub capabilities: HashSet<String>,
        pub retry_policy: RetryPolicy,
    }

    #[derive(Clone, Copy, PartialEq, Eq, Hash)]
    pub struct AgentId(usize);

    impl std::fmt::Display for AgentId {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "agent-{}", self.0)
        }
    }

    pub enum AgentState {
        Thinking,
        Idle,
        Done,
    }

    #[derive(Debug)]
    pub struct AgentError;

    impl std::fmt::Display for AgentError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "agent error")
        }
    }

    impl std::error::Error for AgentError {}

    pub struct AgentRegistry {
        agents: Vec<AgentInfo>,
    }

    impl AgentRegistry {
        pub fn new() -> Self {
            Self { agents: Vec::new() }
        }

        pub fn register(&mut self, info: AgentInfo) -> AgentId {
            let id = AgentId(self.agents.len());
            self.agents.push(info);
            id
        }

        pub fn transition(&mut self, id: AgentId, _state: AgentState) -> Result<(), AgentError> {
            if id.0 < self.agents.len() {
                Ok(())
            } else {
                Err(AgentError)
            }
        }

        pub fn can(&self, id: AgentId, capability: &str) -> bool {
            self.agents
                .get(id.0)
                .map(|a| a.capabilities.contains(capability))
                .unwrap_or(false)
        }
    }

    impl Default for AgentRegistry {
        fn default() -> Self {
            Self::new()
        }
    }
}
