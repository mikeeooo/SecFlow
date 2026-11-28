import { useState, type ReactNode } from "react";
import { Link, useSearchParams } from "react-router";
import { useAlgorithms, useClusterGraph, useEvaluation } from "../api/hooks";
import { ClusterGraph } from "../components/ClusterGraph";
import { MonitoringDashboard } from "../components/MonitoringDashboard";
import type { Report } from "../api/types";
import { ColumnChart } from "../components/charts";
import { ErrorState, LoadingState } from "../components/States";
import { Card, CardHeader, Chip, KeyValue, PageHeader, Segmented, Stat } from "../components/ui";
import { FAMILY_TEXT, formatDay, formatInt, formatMoney, formatPercent, humanize } from "../lib/format";

function Table({ head, rows, caption }: { head: string[]; rows: ReactNode[][]; caption?: string }) {
  return (
    <div className="overflow-x-auto">
      <table className="tabular w-full min-w-[520px] text-left text-[13px]">
        {caption ? <caption className="sr-only">{caption}</caption> : null}
        <thead className="text-[12px] text-label-2">
          <tr>{head.map((h, i) => <th key={h} className={i ? "py-2 pr-3 text-right font-medium" : "py-2 pr-3 font-medium"}>{h}</th>)}</tr>
        </thead>
        <tbody>
          {rows.map((row, r) => (
            <tr key={r} className="border-t border-separator">
              {row.map((cell, i) => <td key={i} className={i ? "py-2 pr-3 text-right" : "py-2 pr-3"}>{cell}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

const num = (v: unknown, digits = 3) => (typeof v === "number" ? v.toFixed(digits) : "n/a");

type View = "dashboard" | "graph" | "models";

export function AlgorithmsPage() {
  const [params, setParams] = useSearchParams();
  const view = (["dashboard", "graph", "models"].includes(params.get("view") ?? "") ? params.get("view") : "dashboard") as View;
  return (
    <>
      <PageHeader
        title="Algorithm Results"
        subtitle="Monitor the assessments, explore how DBSCAN account-context clusters connect, and review how the frozen models performed."
        action={
          <Segmented
            label="View"
            value={view}
            onChange={(next) => setParams(next === "dashboard" ? {} : { view: next }, { replace: true })}
            options={[
              { value: "dashboard", label: "Dashboard" },
              { value: "graph", label: "Cluster graph" },
              { value: "models", label: "Models & evaluation" },
            ]}
          />
        }
      />
      {view === "dashboard" ? <MonitoringDashboard /> : view === "graph" ? <ClusterGraphView /> : <ModelsAndEvaluation />}
    </>
  );
}

function ClusterGraphView() {
  const [anomalies, setAnomalies] = useState("150");
  const [showCores, setShowCores] = useState(true);
  const { data, error, isPending, isFetching, refetch } = useClusterGraph(Number(anomalies));
  return (
    <Card className="p-4 sm:p-5">
      <div className="mb-4 flex flex-wrap items-end justify-between gap-3">
        <div>
          <h2 className="text-[17px] font-semibold">DBSCAN account-context clusters</h2>
          <p className="text-[13px] text-label-2">How the frozen training clusters connect, and which recorded profiles fall outside all of them.</p>
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <label className="flex items-center gap-2 text-[13px]">
            <input type="checkbox" className="size-4 accent-[var(--sf-accent)]" checked={showCores} onChange={(e) => setShowCores(e.target.checked)} />
            Show core samples
          </label>
          <Segmented label="Anomalies shown" size="sm" value={anomalies} onChange={setAnomalies}
            options={[{ value: "50", label: "50 latest" }, { value: "150", label: "150" }, { value: "300", label: "300" }]} />
        </div>
      </div>
      {isPending ? <LoadingState rows={8} label="Building the cluster graph" /> : error ? (
        <ErrorState error={error} onRetry={() => void refetch()} />
      ) : (
        <div aria-busy={isFetching}>
          <ClusterGraph data={data} showCores={showCores} />
          <details className="mt-4 text-[13px]">
            <summary className="cursor-pointer font-medium">View clusters and anomalies as a table</summary>
            <div className="mt-3 grid gap-5 lg:grid-cols-2">
              <Table caption="Largest clusters" head={["Cluster", "Core samples", "Nearest anomalies"]}
                rows={data.clusters.slice(0, 25).map((c) => [`C${c.cluster}`, formatInt(c.size), c.anomalies])} />
              <Table caption="Anomalies" head={["Request", "Nearest cluster", "Distance", "Decision"]}
                rows={data.nodes.filter((n) => n.kind === "anomaly").slice(0, 40).map((n) => [
                  <Link key={n.id} className="text-accent" to={`/investigations/${encodeURIComponent(n.request_id ?? "")}`}>{n.sender} → {n.recipient}</Link>,
                  `C${n.cluster}`, num(n.distance, 2), n.decision_label ?? "",
                ])} />
            </div>
          </details>
        </div>
      )}
    </Card>
  );
}

function ModelsAndEvaluation() {
  const algorithms = useAlgorithms();
  const evaluation = useEvaluation();
  if (algorithms.isPending) return <Card><LoadingState rows={10} /></Card>;
  if (algorithms.error) return <Card><ErrorState error={algorithms.error} onRetry={() => void algorithms.refetch()} /></Card>;
  const a = algorithms.data;
  const model = a.model as Report | null;
  const policy = a.policy as Report;
  const bins = (a.histogram_bins as number) ?? 20;
  const histogram = (key: string, max = 1) =>
    (a.distributions[key] as number[]).map((count, i) => ({
      label: `${((i * max) / bins).toFixed(max === 1 ? 2 : 0)}–${(((i + 1) * max) / bins).toFixed(max === 1 ? 2 : 0)}`,
      values: { count },
    }));

  return (
    <>
      <Card className="mb-5 p-5">
        <h2 className="text-[17px] font-semibold">How the anomaly-risk index is built</h2>
        <p className="tabular mt-3 rounded-[10px] bg-surface-2 p-3 font-mono text-[13px] break-words">
          risk = 100 × ({policy.weights.isolation_forest} × IF + {policy.weights.lof} × LOF + {policy.weights.dbscan} × DBSCAN + {policy.weights.rules} × rules)
        </p>
        <p className="mt-2 text-[12px] text-label-2">
          {policy.weights_source === "learned"
            ? "Weights learned by exhaustive search on the simplex (step 0.05, LOF ≥ 0.05), maximising average precision on validation days 71–77."
            : "Fixed weights from config/policy.json. Only the Verify and Block thresholds are chosen on validation."}
        </p>
        <div className="mt-4 grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
          <Stat label="Personal accounts" value={`${policy.segments.personal.verify} / ${policy.segments.personal.block}`} hint="Verify / Block thresholds, chosen on validation" tone="verify" />
          <Stat label="Business accounts" value={`${policy.segments.business.verify} / ${policy.segments.business.block}`} hint={`Block also needs ≥ ${policy.min_corroborating_families} corroborating families`} tone="block" />
          <Stat label="Cold-start limit" value={formatMoney(policy.cold_start.max_amount_without_history)} hint="without history, above this or without strong sign-in → Verify" />
          <Stat label="Trusted evidence sources" value={policy.trusted_sources.length} hint={policy.trusted_sources.map(humanize).join(", ")} />
        </div>
        <p className="mt-4 text-[13px] text-label-2">
          Action order: active trusted confirmed-fraud evidence → Block; unsafe assessment → Verify; risk ≥ Block with corroboration → preventive Block;
          risk ≥ Verify → Verify; otherwise Permit. Policy {policy.version} · model {model?.model_version ?? "unavailable"}.
        </p>
      </Card>

      <div className="grid gap-5 lg:grid-cols-2">
        {([
          ["isolation_forest", "Isolation Forest score", 1],
          ["lof", "Local Outlier Factor score", 1],
          ["dbscan", "DBSCAN context score (0 inside a core neighbourhood, 1 at twice eps)", 1],
          ["rules", "Behaviour rules score", 1],
          ["risk_score", "Combined anomaly-risk index", 100],
        ] as const).map(([key, title, max]) => (
          <Card key={key}>
            <CardHeader title={title} subtitle="Distribution across assessed requests in this session" />
            <div className="px-5 pb-5">
              <ColumnChart title={title} columns={histogram(key, max)} series={[{ key: "count", label: "Requests", swatch: "bg-accent" }]} height={130}
                xLabels={[max === 1 ? "0" : "0", max === 1 ? "1" : "100"]} />
            </div>
          </Card>
        ))}
      </div>

      {model ? (
        <div className="mt-5 grid gap-5 lg:grid-cols-2">
          <Card>
            <CardHeader title="DBSCAN account context" subtitle="Fitted offline; new profiles are compared with frozen core samples" />
            <div className="space-y-4 px-5 pb-5">
              <KeyValue items={[
                { label: "eps / min_samples", value: `${num(model.dbscan_context.eps)} / ${model.dbscan_context.min_samples}` },
                { label: "Core samples", value: formatInt(model.dbscan_context.core_samples) },
                { label: "Clusters", value: formatInt(model.dbscan_context.clusters) },
                { label: "Training noise share", value: formatPercent(model.dbscan_context.noise_fraction) },
                ...Object.entries(a.dbscan_context.status_counts as Record<string, number>).map(([k, v]) => ({ label: humanize(k), value: formatInt(v) })),
              ]} />
              <ColumnChart
                title="Profiles outside every core neighbourhood, per day"
                columns={(a.dbscan_context.daily as Report[]).map((d) => ({ label: formatDay(d.date), values: { outside: d.outside_context ?? 0 } }))}
                series={[{ key: "outside", label: "Outside context", swatch: "bg-c-dbscan" }]}
                height={110}
              />
              <p className="text-[12px] text-label-2">{model.dbscan_context.adapter}. A context heuristic, not a fraud verdict.</p>
            </div>
          </Card>
          <Card>
            <CardHeader title="Fitting and selection" subtitle="train.json days 1–70 · validation.json days 71–77 (score mapping, weights) and 78–85 (thresholds)" />
            <div className="space-y-4 px-5 pb-5">
              <KeyValue items={[
                { label: "IsolationForest", value: `${model.selected_parameters.isolation_forest.n_estimators} trees, ${model.selected_parameters.isolation_forest.max_samples} samples` },
                { label: "LocalOutlierFactor", value: `novelty, ${model.selected_parameters.lof.n_neighbors} neighbours, ${humanize(model.selected_parameters.lof.subspace)} subspace` },
                { label: "LOF features", value: (model.lof_features as string[]).length },
                { label: "Fitted rows (IF / LOF)", value: `${formatInt(model.fit_rows.isolation_forest)} / ${formatInt(model.fit_rows.lof)}` },
                { label: "Holdout accounts", value: `${model.holdout_accounts} (never fitted on)` },
                { label: "DBSCAN fit profiles", value: formatInt(model.fit_rows.dbscan) },
                { label: "Score reference rows", value: formatInt(model.fit_rows.score_mapping_reference) },
                { label: "scikit-learn", value: model.sklearn_version },
              ]} />
              {a.training ? (
                <Table
                  caption="Validation average precision by component"
                  head={["Validation average precision", "Value"]}
                  rows={Object.entries(a.training.validation_component_average_precision as Record<string, number>).map(([k, v]) => [humanize(k), num(v)])}
                />
              ) : null}
            </div>
          </Card>
        </div>
      ) : (
        <Card className="mt-5 p-5"><p className="text-block">Model bundle unavailable: {a.model_error}</p></Card>
      )}

      {a.training ? (
        <div className="mt-5 grid gap-5 lg:grid-cols-2">
          <Card>
            <CardHeader title="Threshold tradeoff on validation" subtitle="Best feasible block threshold for each verify threshold (model and rules only)" />
            <div className="px-5 pb-5">
              <Table
                caption="Validation frontier"
                head={["Verify ≥", "Block ≥", "Recall (V or B)", "Recall (B)", "Verify rate", "False blocks"]}
                rows={(a.training.threshold_selection.frontier as Report[]).filter((_, i) => i % 3 === 0).map((r) => [
                  r.verify_threshold, r.block_threshold, formatPercent(r.recall_verify_or_block), formatPercent(r.recall_block_only), formatPercent(r.verify_rate, 2), r.false_blocks,
                ])}
              />
              <p className="mt-3 text-[12px] text-label-2">
                {a.training.threshold_selection.feasible_candidates} of {a.training.threshold_selection.candidates_evaluated} threshold pairs met the goals.
                The initial 35/70 candidate gave a verify rate of {formatPercent(a.training.threshold_selection.initial_candidate?.verify_rate)} on validation.
              </p>
            </div>
          </Card>
          <Card>
            <CardHeader title="Parameter search" subtitle="Bounded candidates, scored on validation behaviour" />
            <div className="space-y-4 px-5 pb-5">
              <Table caption="Isolation Forest search" head={["IsolationForest", "Avg. precision"]}
                rows={(a.training.search.isolation_forest as Report[]).map((r) => [`${r.n_estimators} trees · ${r.max_samples}`, num(r.validation_average_precision)])} />
              <Table caption="LOF search" head={["LOF subspace · neighbours · fit rows", "Avg. precision"]}
                rows={(a.training.search.lof as Report[]).map((r) => {
                  const lof = model?.selected_parameters.lof;
                  const chosen = lof && lof.subspace === r.subspace && lof.n_neighbors === r.n_neighbors && lof.fit_rows_requested === r.fit_rows_requested;
                  return [`${humanize(r.subspace)} · ${r.n_neighbors} · ${r.fit_rows_requested}${chosen ? " (selected)" : ""}`, num(r.validation_average_precision)];
                })} />
              <Table caption="DBSCAN search" head={["DBSCAN min_samples · eps", "Avg. precision", "Fraud outside", "Legit outside"]}
                rows={(a.training.search.dbscan as Report[]).map((r) => [`${r.min_samples} · ${num(r.eps, 2)}`, num(r.validation_average_precision), formatPercent(r.validation_fraud_outside_rate), formatPercent(r.validation_legitimate_outside_rate)])} />
            </div>
          </Card>
        </div>
      ) : null}

      {a.training?.weights_search ? (
        <div className="mt-5 grid gap-5 lg:grid-cols-2">
          <Card>
            <CardHeader title="Learned combination weights" subtitle={`Top candidates of ${formatInt(a.training.weights_search.candidates)} on validation days 71–77 (${formatInt(a.training.weights_search.rows)} requests, ${a.training.weights_search.fraud} fraud)`} />
            <div className="px-5 pb-5">
              <Table caption="Weight search" head={["IF", "LOF", "DBSCAN", "Rules", "Avg. precision"]}
                rows={(a.training.weights_search.top as Report[]).map((r) => [num(r.isolation_forest, 2), num(r.lof, 2), num(r.dbscan, 2), num(r.rules, 2), num(r.average_precision)])} />
            </div>
          </Card>
          <Card>
            <CardHeader title="Thresholds per account segment" subtitle="Grid search on validation with the same goals for each segment" />
            <div className="px-5 pb-5">
              <Table caption="Segment thresholds" head={["Segment", "Requests", "Fraud", "Verify ≥", "Block ≥", "Source"]}
                rows={Object.entries(a.training.segment_thresholds as Record<string, Report>).map(([name, r]) => [
                  humanize(name), formatInt(r.requests), r.fraud,
                  policy.segments[name].verify, policy.segments[name].block,
                  r.fallback ? `global pair (${r.reason})` : `validated (${r.feasible_candidates} feasible pairs)`,
                ])} />
            </div>
          </Card>
        </div>
      ) : null}

      <Card className="mt-5">
        <CardHeader title="Behaviour rules" subtitle="Stable codes, explicit weights; families decide corroboration" />
        <div className="px-5 pb-5">
          <Table caption="Rules" head={["Rule", "Family", "Weight", "Corroborating"]}
            rows={(policy.rules as Report[]).map((r) => [
              <span key={r.code}><code className="text-[12px] font-semibold">{r.code}</code><br /><span className="text-label-2">{r.description}</span></span>,
              FAMILY_TEXT[r.family] ?? r.family, r.weight.toFixed(2), r.corroborating ? "Yes" : "No",
            ])} />
        </div>
      </Card>

      <EvaluationSection query={evaluation} />
    </>
  );
}

function EvaluationSection({ query }: { query: ReturnType<typeof useEvaluation> }) {
  if (query.isPending) return <Card className="mt-5"><LoadingState rows={6} /></Card>;
  if (query.error) return <Card className="mt-5"><ErrorState error={query.error} title="No evaluation report" onRetry={() => void query.refetch()} /></Card>;
  const e = query.data;
  const full = e.combined_policy;
  const modelOnly = e.combined_policy_without_evidence;
  const view = (label: string, b: Report) => [label, formatPercent(b.precision), formatPercent(b.recall), num(b.f1), formatPercent(b.false_positive_rate, 2),
    `${b.confusion_matrix.true_positive} / ${b.confusion_matrix.false_positive} / ${b.confusion_matrix.false_negative} / ${b.confusion_matrix.true_negative}`];
  return (
    <section className="mt-8" aria-labelledby="eval-title">
      <h2 id="eval-title" className="font-display text-[24px] font-bold tracking-[-0.02em]">Evaluation on the final test period</h2>
      <p className="mt-1 max-w-3xl text-[14px] text-label-2">{e.scope}</p>
      <Card className="mt-4 grid grid-cols-2 gap-6 p-5 md:grid-cols-4">
        <Stat label="Fraud flagged (Verify or Block)" value={formatPercent(full.verify_or_block.recall)} hint={`${e.test_fraud} fraud among ${formatInt(e.test_requests)} requests`} />
        <Stat label="Fraud blocked" value={formatPercent(full.block_only.recall)} hint={`block precision ${formatPercent(full.block_only.precision)} over ${full.predicted_blocks} blocks`} tone="block" />
        <Stat label="Verify rate" value={formatPercent(full.verify_rate, 2)} tone="verify" hint={`legitimate false blocks ${formatPercent(full.legitimate_false_block_rate, 3)}`} />
        <Stat label="p95 latency" value={`${e.latency_ms.p95} ms`} hint={`${e.latency_ms.requests_timed} single requests, in process`} />
      </Card>
      {e.confidence_intervals ? (
        <p className="mt-3 text-[13px] text-label-2">
          95% intervals: recall {formatPercent(e.confidence_intervals.recall_verify_or_block[0])}–{formatPercent(e.confidence_intervals.recall_verify_or_block[1])},
          precision {formatPercent(e.confidence_intervals.precision_verify_or_block[0])}–{formatPercent(e.confidence_intervals.precision_verify_or_block[1])},
          verify rate {formatPercent(e.confidence_intervals.verify_rate[0], 2)}–{formatPercent(e.confidence_intervals.verify_rate[1], 2)} ({e.confidence_intervals.method}).
        </p>
      ) : null}
      {e.score_bands ? <ScoreBandsCard bands={e.score_bands} verify={e.operating_point.verify} block={e.operating_point.block} /> : null}
      <div className="mt-5 grid gap-5 lg:grid-cols-2">
        <Card>
          <CardHeader title="Combined policy" subtitle="TP / FP / FN / TN; positive = Verify or Block, or Block only" />
          <div className="px-5 pb-5">
            <Table caption="Combined policy" head={["View", "Precision", "Recall", "F1", "FPR", "Confusion"]}
              rows={[
                view("With evidence · V or B", full.verify_or_block), view("With evidence · Block", full.block_only),
                view("Model + rules · V or B", modelOnly.verify_or_block), view("Model + rules · Block", modelOnly.block_only),
              ]} />
            <p className="mt-3 text-[12px] text-label-2">
              Evidence blocks: {e.evidence_channel.evidence_blocks} ({e.evidence_channel.evidence_blocks_on_fraud} fraud, {e.evidence_channel.evidence_blocks_on_legitimate} legitimate).
              Without the evidence channel, model and rules flag {formatPercent(e.evidence_channel.model_only_recall_on_fraud_without_evidence)} of fraud that had no evidence.
            </p>
          </div>
        </Card>
        <Card>
          <CardHeader title="Each component alone" subtitle="Flag thresholds fixed on validation (top 5% budget)" />
          <div className="px-5 pb-5">
            <Table caption="Components" head={["Component", "Precision", "Recall", "FPR", "Avg. precision", "ROC AUC"]}
              rows={["isolation_forest", "lof", "dbscan", "rules"].map((k) => {
                const c = e.components[k];
                return [humanize(k), formatPercent(c.precision), formatPercent(c.recall), formatPercent(c.false_positive_rate, 2), num(c.average_precision), num(c.roc_auc)];
              })} />
          </div>
        </Card>
        <Card>
          <CardHeader title="Recall by fraud type" subtitle="Share of each fraud type's test payments sent to Verify or Block" />
          <div className="px-5 pb-5">
            <Table caption="Fraud types" head={["Fraud type", "Payments", "V or B", "Block", "Evidence block"]}
              rows={Object.entries((e.per_fraud_type ?? {}) as Record<string, Report>).map(([k, r]) => [
                <span key={k}>{humanize(k)}{r.test_only ? <> <Chip tone="verify">test only</Chip></> : null}</span>,
                formatInt(r.requests), formatPercent(r.recall_verify_or_block), formatPercent(r.recall_block_only), formatPercent(r.evidence_block_rate),
              ])} />
          </div>
        </Card>
        <Card>
          <CardHeader title="Unusual but legitimate" subtitle="Share of each anomaly type sent to Verify or Block (legitimate payments)" />
          <div className="px-5 pb-5">
            <Table caption="Anomaly types" head={["Anomaly type", "Payments", "V or B", "Block"]}
              rows={Object.entries((e.per_anomaly_type ?? {}) as Record<string, Report>).map(([k, r]) => [
                humanize(k), formatInt(r.requests), formatPercent(r.verify_or_block_rate), formatPercent(r.block_rate),
              ])} />
          </div>
        </Card>
        <Card>
          <CardHeader title="Subsets and goals" />
          <div className="space-y-4 px-5 pb-5">
            <Table caption="Subsets" head={["Subset", "Requests", "Fraud recall", "Legit review", "Legit false block"]}
              rows={[
                ["Unusual legitimate", e.unusual_legitimate], ["Cold-start accounts", e.cold_start_accounts], ["Existing accounts", e.existing_accounts],
                ...(e.holdout_accounts ? [["Holdout accounts (never fitted)", e.holdout_accounts]] : []),
                ...(e.drift_billers ? [["Payments to drift billers", e.drift_billers]] : []),
                ...(e.unseen_scenario ? [["Test-only types: QR tampering, invoice redirect", e.unseen_scenario]] : []),
              ].map(([label, s]) => [
                label as string, formatInt((s as Report).requests), formatPercent((s as Report).verify_or_block?.recall), formatPercent((s as Report).legitimate_review_rate, 2), formatPercent((s as Report).legitimate_false_block_rate, 3),
              ])} />
            <ul className="space-y-1.5 text-[13px]">
              {(e.goal_assessment as Report[]).map((g) => (
                <li key={g.goal} className="flex flex-wrap items-center gap-2">
                  <Chip tone={g.met === true ? "permit" : g.met === false ? "verify" : "neutral"}>{g.met === true ? "Met" : g.met === false ? "Not met" : "Not established"}</Chip>
                  <span>{g.goal}: measured {typeof g.measured === "number" ? g.measured.toFixed(4) : "n/a"}</span>
                </li>
              ))}
            </ul>
            <p className="text-[12px] text-label-2">Model {e.model_version} · generated {e.generated_at}. Goals are targets to assess, not promises.</p>
          </div>
        </Card>
      </div>
    </section>
  );
}

/** The yield of the risk percentage: observed fraud rate per score band on labelled test data. */
function ScoreBandsCard({ bands, verify, block }: { bands: Report; verify: number; block: number }) {
  const rows = bands.bands as Report[];
  const zone = (low: number) => (low >= block ? "Block zone" : low >= verify ? "Verify zone" : "Permit zone");
  return (
    <Card className="mt-5">
      <CardHeader
        title="What the risk percentage is worth"
        subtitle={`Observed fraud rate per score band over ${formatInt(bands.requests)} labelled test requests (base rate ${formatPercent(bands.base_fraud_rate, 2)})`}
      />
      <div className="space-y-4 px-5 pb-5">
        <p className="max-w-3xl text-[14px]">
          The risk score is an anomaly ranking from 0 to 100, not a probability. Its real yield is read from this table: the higher the band,
          the more often requests in it were really fraud. <em>Recall ≥ band</em> is the share of all fraud caught if every request at or above
          the band&apos;s lower edge were flagged; <em>precision ≥ band</em> is how many of those flags were real fraud.
        </p>
        <ul className="space-y-1.5" aria-label="Fraud rate per score band">
          {rows.map((b) => (
            <li key={b.low} className="grid grid-cols-[64px_minmax(0,1fr)_64px] items-center gap-3 text-[13px] sm:grid-cols-[72px_minmax(0,1fr)_72px_110px]">
              <span className="tabular text-label-2">{b.low}–{b.high}</span>
              <div className="h-3 overflow-hidden rounded-full bg-surface-2">
                <div className={b.low >= block ? "h-full bg-block-strong" : b.low >= verify ? "h-full bg-verify" : "h-full bg-permit"}
                  style={{ width: `${Math.max(b.fraud_rate ? 1 : 0, (b.fraud_rate ?? 0) * 100)}%` }} />
              </div>
              <span className="tabular text-right font-semibold">{formatPercent(b.fraud_rate)}</span>
              <span className="hidden text-[12px] text-label-2 sm:block">{zone(b.low)}</span>
            </li>
          ))}
        </ul>
        <Table caption="Score bands" head={["Score band", "Requests", "Fraud", "Fraud rate (95% CI)", "Lift", "Recall ≥ band", "Precision ≥ band", "Flagged ≥ band"]}
          rows={rows.map((b) => [
            `${b.low}–${b.high}`, formatInt(b.requests), formatInt(b.fraud),
            `${formatPercent(b.fraud_rate)}${b.fraud_rate_95ci ? ` (${formatPercent(b.fraud_rate_95ci[0])}–${formatPercent(b.fraud_rate_95ci[1])})` : ""}`,
            b.lift == null ? "n/a" : `${b.lift.toFixed(1)}×`, formatPercent(b.recall_at_or_above), formatPercent(b.precision_at_or_above),
            formatPercent(b.flag_rate_at_or_above, 2),
          ])} />
        <p className="text-[12px] text-label-2">{bands.note}</p>
      </div>
    </Card>
  );
}
