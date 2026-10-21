//! Fixed-pair reranker health gate shared by native load and doctor; v1 §7.5.
use super::Error;
use crate::models::{self, Reranker};

pub(crate) const QUERY: &str = "What is the capital of France?";
pub(crate) const DOCUMENTS: [&str; 3] = [
    "The capital of France is Paris.",
    "Bananas are yellow fruit.",
    "France is a country in Europe.",
];

/// Validates complete probability outputs and the specified relevance/variance gates.
/// The filename is diagnostic identity, never a prompt or untrusted provider body.
pub(crate) fn validate(scores: &[f32], file: &str) -> Result<[f32; 3], Error> {
    let failed = || Error::Smoke { file: file.into() };
    let scores: [f32; 3] = scores.try_into().map_err(|_| failed())?;
    if scores
        .iter()
        .any(|s| !s.is_finite() || !(0.0..=1.0).contains(s))
        || scores[0] <= 0.5
        || scores[1] >= 0.2
        || scores.iter().copied().fold(f32::NEG_INFINITY, f32::max)
            - scores.iter().copied().fold(f32::INFINITY, f32::min)
            <= 0.02
    {
        return Err(failed());
    }
    Ok(scores)
}
/// Runs the fixed public pairs through a prepared provider, suitable for doctor.
pub(crate) async fn run(provider: &dyn Reranker, file: &str) -> models::Result<[f32; 3]> {
    Ok(validate(&provider.score(QUERY, &DOCUMENTS).await?, file)?)
}
#[cfg(test)]
mod tests {
    use super::*;
    struct Mock(Vec<f32>);
    #[async_trait::async_trait]
    impl Reranker for Mock {
        fn id(&self) -> &str {
            "mock"
        }
        fn max_doc_chars(&self) -> usize {
            12000
        }
        async fn score(&self, query: &str, docs: &[&str]) -> models::Result<Vec<f32>> {
            assert_eq!(query, QUERY);
            assert_eq!(docs, DOCUMENTS);
            Ok(self.0.clone())
        }
    }
    #[tokio::test]
    async fn broken_scorers_fail_with_model_identity() {
        assert_eq!(
            run(&Mock(vec![0.9, 0.1, 0.4]), "healthy.gguf")
                .await
                .unwrap(),
            [0.9, 0.1, 0.4]
        );
        for scores in [
            vec![0.5; 3],
            vec![0.1; 3],
            vec![0.5, 0.1, 0.3],
            vec![0.9, 0.2, 0.4],
            vec![0.9, 0.1],
            vec![0.9, 0.1, f32::NAN],
            vec![1.1, 0.1, 0.3],
        ] {
            let error = run(&Mock(scores), "broken.gguf").await.unwrap_err();
            assert!(
                matches!(error,models::Error::Rerank(Error::Smoke{ref file}) if file=="broken.gguf")
            );
        }
    }
}
