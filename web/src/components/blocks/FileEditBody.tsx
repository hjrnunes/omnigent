// Expanded view of a file-edit tool call: the change as a diff, with the
// tool's reported failure, or output that does not confirm the edit, above it.

import { useMemo } from "react";
import { getFileEditDiff } from "@/lib/toolEditDiff";
import { type ToolCall, editApplied } from "@/lib/toolResultSummary";
import { editErrorMessage } from "@/lib/toolValue";
import { CodePanel, OutputSection, ToolErrorBox } from "./ToolCodePanel";
import { ToolValueView } from "./ToolValueView";

const EMPTY_FAILURE = "The edit failed with no output.";

export function FileEditBody({ call }: { call: ToolCall }) {
  const { name, args, output, state } = call;
  // Before output arrives, the card shows the pending or cancelled state below instead.
  const awaitingOutput = output === null && state !== "output-available";
  const diff = useMemo(() => getFileEditDiff(name, args), [name, args]);
  const error = useMemo(() => {
    if (output === null) return null;
    // Defensive: the transcript sets output-error only without output, but a direct
    // caller's failed output is its error even without a recognized prefix.
    const failure = state === "output-error" ? output.trim() || EMPTY_FAILURE : null;
    return editErrorMessage(output) ?? failure;
  }, [output, state]);
  // The diff is what was requested; show output that does not say it applied.
  const unconfirmed =
    output !== null && error === null && output.trim() !== "" && !editApplied(name, output);
  const outputPanel = output !== null && <OutputSection output={output} language="text" />;
  return (
    <div className="space-y-2">
      {error !== null && <ToolErrorBox>{error}</ToolErrorBox>}
      {diff === null ? (
        <>
          {!awaitingOutput && (
            <p className="text-muted-foreground text-sm">No diff recorded for this edit.</p>
          )}
          {Object.keys(args).length > 0 && <ToolValueView value={args} />}
          {output !== null && error === null && output.trim() !== "" && outputPanel}
        </>
      ) : (
        <>
          {unconfirmed && outputPanel}
          <OutputSection
            output={diff.text}
            renderPanel={(text) => (
              <CodePanel
                title="Diff"
                text={text}
                copyText={diff.text}
                copyLabel="Copy diff"
                language="diff"
              />
            )}
          />
        </>
      )}
    </div>
  );
}
