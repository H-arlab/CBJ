import { Component, type ErrorInfo, type ReactNode } from 'react';

/**
 * Last-resort error boundary so a single crashing component doesn't
 * blank the whole workspace. Renders a minimal recovery UI with two
 * exits: "Try again" (re-mount the children) and "Reset workspace"
 * (clear the persisted Zustand state and reload).
 *
 * The accent / neon palette is mirrored inline so this component
 * keeps working even when stylesheets fail to load.
 */
interface Props {
  children: ReactNode;
}

interface State {
  error: Error | null;
  info: ErrorInfo | null;
}

export default class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null, info: null };

  static getDerivedStateFromError(error: Error): Partial<State> {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    console.error('[ErrorBoundary]', error, info);
    this.setState({ info });
  }

  handleRetry = (): void => {
    this.setState({ error: null, info: null });
  };

  handleReset = (): void => {
    try {
      localStorage.removeItem('hw_page_v1');
    } catch {
      /* ignore */
    }
    const url = new URL(window.location.href);
    url.searchParams.set('reset', 'true');
    window.location.href = url.toString();
  };

  render(): ReactNode {
    if (!this.state.error) return this.props.children;

    const message = this.state.error.message || String(this.state.error);
    const stack = this.state.error.stack || '';
    const componentStack = this.state.info?.componentStack || '';

    return (
      <div
        style={{
          minHeight: '100vh',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          background: '#0B0E2E',
          color: '#fff',
          fontFamily: "'Pretendard', system-ui, sans-serif",
          padding: 32,
        }}
      >
        <div style={{ maxWidth: 720, width: '100%' }}>
          <div
            style={{
              fontSize: 11,
              letterSpacing: '.22em',
              textTransform: 'uppercase',
              color: '#F09708',
              marginBottom: 12,
            }}
          >
            Workspace error
          </div>
          <h1 style={{ fontSize: 24, fontWeight: 600, margin: '0 0 16px' }}>
            Something rendered into an unexpected state.
          </h1>
          <p style={{ opacity: 0.8, lineHeight: 1.6, margin: '0 0 24px' }}>
            The workspace caught a crash before it blanked the page.
            "Try again" re-mounts the components — fixes most transient
            issues. If it still fails, "Reset workspace" clears the
            saved state and starts over (your uploaded CSVs are kept).
          </p>
          <div style={{ display: 'flex', gap: 12, marginBottom: 24 }}>
            <button
              onClick={this.handleRetry}
              style={{
                padding: '10px 18px',
                background: '#F09708',
                color: '#0B0E2E',
                border: 0,
                borderRadius: 4,
                fontWeight: 600,
                cursor: 'pointer',
              }}
            >
              Try again
            </button>
            <button
              onClick={this.handleReset}
              style={{
                padding: '10px 18px',
                background: 'transparent',
                color: '#fff',
                border: '1px solid rgba(255,255,255,0.3)',
                borderRadius: 4,
                cursor: 'pointer',
              }}
            >
              Reset workspace
            </button>
          </div>
          <details
            style={{
              fontFamily: "'JetBrains Mono', monospace",
              fontSize: 12,
              opacity: 0.7,
            }}
          >
            <summary style={{ cursor: 'pointer', marginBottom: 8 }}>
              Diagnostic info (click to expand)
            </summary>
            <pre
              style={{
                whiteSpace: 'pre-wrap',
                wordBreak: 'break-word',
                padding: 12,
                background: 'rgba(255,255,255,0.05)',
                borderRadius: 4,
                margin: 0,
              }}
            >
              {message}
              {stack ? `\n\n${stack}` : ''}
              {componentStack ? `\n\nComponent stack:${componentStack}` : ''}
            </pre>
          </details>
        </div>
      </div>
    );
  }
}
