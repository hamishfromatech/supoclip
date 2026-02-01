"use client";

import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Separator } from "@/components/ui/separator";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Skeleton } from "@/components/ui/skeleton";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useSession } from "@/lib/auth-client";
import Link from "next/link";
import { ArrowLeft, Cpu, Settings2, CheckCircle, AlertCircle, Info, Server, Key } from "lucide-react";

export default function AISettingsPage() {
  const [aiProvider, setAiProvider] = useState("local");
  const [aiModel, setAiModel] = useState("llama3");
  const [aiApiKey, setAiApiKey] = useState("");
  const [aiBaseUrl, setAiBaseUrl] = useState("http://host.docker.internal:11434/v1");
  const [whisperModel, setWhisperModel] = useState("medium");
  const [isLoading, setIsLoading] = useState(false);
  const [isFetching, setIsFetching] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);
  const { data: session, isPending } = useSession();

  // Load AI settings
  useEffect(() => {
    const loadSettings = async () => {
      if (!session?.user?.id) return;

      setIsFetching(true);
      try {
        const response = await fetch('/api/ai-settings');
        if (response.ok) {
          const data = await response.json();
          setAiProvider(data.aiProvider || "local");
          setAiModel(data.aiModel || "llama3");
          setAiApiKey(data.aiApiKey || "");
          setAiBaseUrl(data.aiBaseUrl || "http://host.docker.internal:11434/v1");
          setWhisperModel(data.whisperModel || "medium");
        }
      } catch (error) {
        console.error('Failed to load AI settings:', error);
      } finally {
        setIsFetching(false);
      }
    };

    loadSettings();
  }, [session?.user?.id]);

  const handleSaveSettings = async () => {
    setIsLoading(true);
    setError(null);
    setSuccess(false);

    try {
      const response = await fetch('/api/ai-settings', {
        method: 'PATCH',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          aiProvider,
          aiModel,
          aiApiKey,
          aiBaseUrl,
          whisperModel,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to save settings');
      }

      setSuccess(true);
      setTimeout(() => setSuccess(false), 3000);
    } catch (error) {
      console.error('Error saving settings:', error);
      setError(error instanceof Error ? error.message : 'Failed to save settings');
    } finally {
      setIsLoading(false);
    }
  };

  if (isPending || isFetching) {
    return (
      <div className="min-h-screen bg-white flex items-center justify-center p-4">
        <div className="space-y-4">
          <Skeleton className="h-4 w-32 mx-auto" />
          <Skeleton className="h-4 w-48 mx-auto" />
          <Skeleton className="h-4 w-24 mx-auto" />
        </div>
      </div>
    );
  }

  if (!session?.user) {
    return (
      <div className="min-h-screen bg-white">
        <div className="max-w-4xl mx-auto px-4 py-24">
          <div className="text-center">
            <h1 className="text-3xl font-bold text-black mb-4">
              Sign In Required
            </h1>
            <p className="text-gray-600 mb-8">
              You need to sign in to access AI settings
            </p>
            <Link href="/sign-in">
              <Button size="lg">Sign In</Button>
            </Link>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-white">
      {/* Header */}
      <div className="border-b bg-white">
        <div className="max-w-4xl mx-auto px-4 py-4">
          <div className="flex justify-between items-center">
            <Link href="/settings" className="flex items-center gap-3 hover:opacity-80 transition-opacity cursor-pointer">
              <ArrowLeft className="w-5 h-5 text-black" />
              <h1 className="text-xl font-bold text-black">AI Settings</h1>
            </Link>

            <div className="flex items-center gap-3">
              <Avatar className="w-8 h-8">
                <AvatarImage src={session.user.image || ""} />
                <AvatarFallback className="bg-gray-100 text-black text-sm">
                  {session.user.name?.charAt(0) || session.user.email?.charAt(0) || "U"}
                </AvatarFallback>
              </Avatar>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-4xl mx-auto px-4 py-16">
        <div className="max-w-xl mx-auto">
          <div className="mb-8">
            <div className="flex items-center gap-2 mb-2">
              <Cpu className="w-6 h-6 text-black" />
              <h2 className="text-2xl font-bold text-black">
                AI Configuration
              </h2>
            </div>
            <p className="text-gray-600">
              Configure your AI provider and model preferences for video processing
            </p>
          </div>

          <Separator className="my-8" />

          <div className="space-y-8">
            {/* AI Provider Selection */}
            <div className="space-y-6">
              <div>
                <h3 className="text-lg font-semibold text-black mb-1">
                  AI Provider
                </h3>
                <p className="text-sm text-gray-600">
                  Select your AI provider for video content analysis
                </p>
              </div>

              <div className="space-y-4">
                <Label className="text-sm font-medium text-black flex items-center gap-2">
                  <Server className="w-4 h-4" />
                  Provider
                </Label>
                <Select value={aiProvider} onValueChange={setAiProvider} disabled={isLoading}>
                  <SelectTrigger className="w-full">
                    <SelectValue placeholder="Select provider" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="local">
                      <div className="flex items-center gap-2">
                        <Cpu className="w-4 h-4" />
                        Local (Ollama/LM Studio)
                      </div>
                    </SelectItem>
                    <SelectItem value="openai">
                      <div className="flex items-center gap-2">
                        <Key className="w-4 h-4" />
                        OpenAI
                      </div>
                    </SelectItem>
                    <SelectItem value="google">
                      Google
                    </SelectItem>
                    <SelectItem value="anthropic">
                      Anthropic
                    </SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* Local Provider Configuration */}
              {aiProvider === "local" && (
                <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 space-y-4">
                  <div className="flex items-start gap-3">
                    <Info className="w-5 h-5 text-blue-600 mt-0.5" />
                    <div className="flex-1">
                      <h4 className="font-medium text-black mb-1">Local AI Configuration</h4>
                      <p className="text-sm text-gray-700">
                        Configure your local AI provider (Ollama or LM Studio) running on your machine.
                      </p>
                    </div>
                  </div>

                  <div className="space-y-2">
                    <Label className="text-sm font-medium text-black">
                      Base URL
                    </Label>
                    <Input
                      type="text"
                      value={aiBaseUrl}
                      onChange={(e) => setAiBaseUrl(e.target.value)}
                      disabled={isLoading}
                      placeholder="http://host.docker.internal:11434/v1"
                      className="font-mono text-sm"
                    />
                    <p className="text-xs text-gray-600">
                      Ollama default: http://host.docker.internal:11434/v1
                    </p>
                    <p className="text-xs text-gray-600">
                      LM Studio default: http://host.docker.internal:1234/v1
                    </p>
                  </div>

                  <div className="space-y-2">
                    <Label className="text-sm font-medium text-black">
                      Model Name
                    </Label>
                    <Input
                      type="text"
                      value={aiModel}
                      onChange={(e) => setAiModel(e.target.value)}
                      disabled={isLoading}
                      placeholder="llama3"
                      className="font-mono text-sm"
                    />
                    <p className="text-xs text-gray-600">
                      Examples: llama3, mistral, codellama, deepseek-coder
                    </p>
                  </div>
                </div>
              )}

              {/* Cloud Provider Configuration */}
              {aiProvider !== "local" && (
                <div className="bg-purple-50 border border-purple-200 rounded-lg p-4 space-y-4">
                  <div className="flex items-start gap-3">
                    <Key className="w-5 h-5 text-purple-600 mt-0.5" />
                    <div className="flex-1">
                      <h4 className="font-medium text-black mb-1">{aiProvider.charAt(0).toUpperCase() + aiProvider.slice(1)} API Configuration</h4>
                      <p className="text-sm text-gray-700">
                        Enter your API key and select a model from {aiProvider.charAt(0).toUpperCase() + aiProvider.slice(1)}.
                      </p>
                    </div>
                  </div>

                  <div className="space-y-2">
                    <Label className="text-sm font-medium text-black">
                      API Key
                    </Label>
                    <Input
                      type="password"
                      value={aiApiKey}
                      onChange={(e) => setAiApiKey(e.target.value)}
                      disabled={isLoading}
                      placeholder="Enter your API key"
                      className="font-mono text-sm"
                    />
                    <p className="text-xs text-gray-600">
                      Your API key is encrypted and stored securely
                    </p>
                  </div>

                  <div className="space-y-2">
                    <Label className="text-sm font-medium text-black">
                      Model
                    </Label>
                    <Input
                      type="text"
                      value={aiModel}
                      onChange={(e) => setAiModel(e.target.value)}
                      disabled={isLoading}
                      placeholder="gpt-4"
                      className="font-mono text-sm"
                    />
                    <p className="text-xs text-gray-600">
                      {aiProvider === "openai" && "Examples: gpt-4, gpt-4-turbo, gpt-3.5-turbo"}
                      {aiProvider === "google" && "Examples: gemini-pro, gemini-1.5-pro"}
                      {aiProvider === "anthropic" && "Examples: claude-3-opus, claude-3-sonnet, claude-3-haiku"}
                    </p>
                  </div>
                </div>
              )}
            </div>

            <Separator className="my-8" />

            {/* Whisper Model Selection */}
            <div className="space-y-6">
              <div>
                <h3 className="text-lg font-semibold text-black mb-1">
                  Whisper Model
                </h3>
                <p className="text-sm text-gray-600">
                  Select the model size for video transcription
                </p>
              </div>

              <div className="space-y-2">
                <Label className="text-sm font-medium text-black">
                  Model Size
                </Label>
                <Select value={whisperModel} onValueChange={setWhisperModel} disabled={isLoading}>
                  <SelectTrigger className="w-full">
                    <SelectValue placeholder="Select model size" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="tiny">Tiny (Fastest, Less Accurate)</SelectItem>
                    <SelectItem value="base">Base (Fast)</SelectItem>
                    <SelectItem value="small">Small (Balanced)</SelectItem>
                    <SelectItem value="medium">Medium (Recommended)</SelectItem>
                    <SelectItem value="large">Large (Most Accurate, Slowest)</SelectItem>
                  </SelectContent>
                </Select>
                <p className="text-xs text-gray-600">
                  Larger models provide more accurate transcriptions but take longer to process
                </p>
              </div>

              <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
                <div className="flex items-start gap-3">
                  <AlertCircle className="w-5 h-5 text-yellow-600 mt-0.5" />
                  <div className="flex-1">
                    <h4 className="font-medium text-black mb-1">Performance Note</h4>
                    <p className="text-sm text-gray-700">
                      The Whisper transcription model runs locally on the server. Larger models require more memory and processing time but provide better accuracy. We recommend starting with "medium" as a balance.
                    </p>
                  </div>
                </div>
              </div>
            </div>

            {/* Success/Error Messages */}
            {success && (
              <Alert className="border-green-200 bg-green-50">
                <CheckCircle className="h-4 w-4 text-green-500" />
                <AlertDescription className="text-sm text-green-700">
                  AI settings saved successfully!
                </AlertDescription>
              </Alert>
            )}

            {error && (
              <Alert className="border-red-200 bg-red-50">
                <AlertCircle className="h-4 w-4 text-red-500" />
                <AlertDescription className="text-sm text-red-700">
                  {error}
                </AlertDescription>
              </Alert>
            )}

            {/* Save Button */}
            <Button
              onClick={handleSaveSettings}
              disabled={isLoading}
              className="w-full h-11"
            >
              {isLoading ? "Saving..." : "Save Settings"}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}