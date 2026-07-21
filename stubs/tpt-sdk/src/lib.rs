pub mod keystone {
    #[derive(Debug)]
    pub enum KeystoneError {
        Connection(String),
    }

    impl std::fmt::Display for KeystoneError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                KeystoneError::Connection(s) => write!(f, "connection error: {s}"),
            }
        }
    }

    impl std::error::Error for KeystoneError {}

    #[derive(Clone, Debug, PartialEq)]
    pub enum Value {
        Text(String),
        Int(i64),
        Float(f64),
        Null,
    }

    #[derive(Clone, Debug)]
    pub struct Row {
        column_names: Vec<String>,
        values: Vec<Option<Vec<u8>>>,
    }

    impl Row {
        pub fn new<S: AsRef<str>>(column_names: &[S], values: &[Option<Vec<u8>>]) -> Self {
            Self {
                column_names: column_names
                    .iter()
                    .map(|s| s.as_ref().to_string())
                    .collect(),
                values: values.to_vec(),
            }
        }

        pub fn column_names(&self) -> &[String] {
            &self.column_names
        }

        pub fn get_value(&self, idx: usize) -> Value {
            let Some(Some(bytes)) = self.values.get(idx) else {
                return Value::Null;
            };
            let Ok(s) = std::str::from_utf8(bytes) else {
                return Value::Null;
            };
            if let Ok(i) = s.parse::<i64>() {
                return Value::Int(i);
            }
            if let Ok(f) = s.parse::<f64>() {
                return Value::Float(f);
            }
            Value::Text(s.to_string())
        }
    }

    #[derive(Clone, Debug)]
    pub struct QueryResult {
        pub columns: Vec<String>,
        pub rows: Vec<Row>,
    }

    impl QueryResult {
        pub fn new(columns: Vec<String>, rows: Vec<Row>, _meta: Option<()>) -> Self {
            Self { columns, rows }
        }
    }

    pub struct KeystoneClient;

    impl KeystoneClient {
        pub async fn connect(_addr: &str) -> Result<Self, KeystoneError> {
            Err(KeystoneError::Connection("stub implementation".to_string()))
        }

        pub async fn query_params(
            &mut self,
            _sql: &str,
            _params: &[Value],
        ) -> Result<QueryResult, KeystoneError> {
            Err(KeystoneError::Connection("stub implementation".to_string()))
        }
    }
}

pub mod query_builder {
    use super::keystone::Value;

    pub trait Table {
        const NAME: &'static str;
        const COLUMNS: &'static [&'static str];
    }

    pub enum Order {
        Asc,
        Desc,
    }

    pub struct QueryBuilder<T: Table> {
        filters: Vec<(String, Value)>,
        order: Option<(String, Order)>,
        limit: Option<usize>,
        _marker: std::marker::PhantomData<T>,
    }

    impl<T: Table> QueryBuilder<T> {
        pub fn new() -> Self {
            Self {
                filters: Vec::new(),
                order: None,
                limit: None,
                _marker: std::marker::PhantomData,
            }
        }

        pub fn filter_eq(mut self, col: &str, value: Value) -> Self {
            self.filters.push((col.to_string(), value));
            self
        }

        pub fn order_by(mut self, col: &str, order: Order) -> Self {
            self.order = Some((col.to_string(), order));
            self
        }

        pub fn limit(mut self, n: usize) -> Self {
            self.limit = Some(n);
            self
        }

        pub fn build(self) -> (String, Vec<Value>) {
            let QueryBuilder {
                filters,
                order,
                limit,
                ..
            } = self;
            let mut sql = format!("SELECT * FROM {}", T::NAME);
            let mut params = Vec::new();
            if !filters.is_empty() {
                sql.push_str(" WHERE ");
                for (i, (col, val)) in filters.into_iter().enumerate() {
                    if i > 0 {
                        sql.push_str(" AND ");
                    }
                    sql.push_str(&format!("{col} = ${}", i + 1));
                    params.push(val);
                }
            }
            if let Some((col, ord)) = order {
                let dir = match ord {
                    Order::Asc => "ASC",
                    Order::Desc => "DESC",
                };
                sql.push_str(&format!(" ORDER BY {col} {dir}"));
            }
            if let Some(n) = limit {
                sql.push_str(&format!(" LIMIT {n}"));
            }
            (sql, params)
        }
    }

    impl<T: Table> Default for QueryBuilder<T> {
        fn default() -> Self {
            Self::new()
        }
    }
}
