import { createFileRoute } from "@tanstack/react-router";
import { ThemeProvider } from "@/lib/theme";
import { ParticleField } from "@/components/si/ParticleField";
import { Header, Footer, MagneticLink } from "@/components/si/Chrome";
import { Hero } from "@/components/si/Hero";
import { Pipeline } from "@/components/si/Pipeline";
import { Capabilities } from "@/components/si/Capabilities";
import { ResultPreview } from "@/components/si/ResultPreview";

export const Route = createFileRoute("/")({
  component: Index,
});

function Index() {
  return (
    <ThemeProvider>
      <div className="relative min-h-screen">
        <ParticleField />
        <Header />
        <main>
          <Hero />
          <Pipeline />
          <Capabilities />
          <ResultPreview />

          <section id="analyze" className="mx-auto max-w-6xl px-6 pb-40">
            <div className="border border-border bg-background/60 px-8 py-20 text-center backdrop-blur-sm">
              <p className="label-mono">Next step</p>
              <h2 className="mx-auto mt-6 max-w-lg text-3xl tracking-[-0.02em] sm:text-4xl">
                Ready to analyze a signal?
              </h2>
              <div className="mt-10 flex justify-center">
                <MagneticLink href="/analyze">Analyze a signal</MagneticLink>
              </div>
            </div>
          </section>
        </main>
        <Footer />
      </div>
    </ThemeProvider>
  );
}
