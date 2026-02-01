import { NextRequest, NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import { headers } from "next/headers";
import prisma from "@/lib/prisma";

// GET /api/ai-settings - Get user AI settings
export async function GET(request: NextRequest) {
  try {
    const session = await auth.api.getSession({
      headers: await headers(),
    });

    if (!session?.user?.id) {
      return NextResponse.json(
        { error: "Unauthorized" },
        { status: 401 }
      );
    }

    const user = await prisma.user.findUnique({
      where: { id: session.user.id },
      select: {
        ai_provider: true,
        ai_model: true,
        ai_api_key: true,
        ai_base_url: true,
        whisper_model: true,
      },
    });

    if (!user) {
      return NextResponse.json(
        { error: "User not found" },
        { status: 404 }
      );
    }

    return NextResponse.json({
      aiProvider: user.ai_provider || "local",
      aiModel: user.ai_model || "llama3",
      aiApiKey: user.ai_api_key || "",
      aiBaseUrl: user.ai_base_url || "http://host.docker.internal:11434/v1",
      whisperModel: user.whisper_model || "medium",
    });
  } catch (error) {
    console.error("Error fetching AI settings:", error);
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    );
  }
}

// PATCH /api/ai-settings - Update user AI settings
export async function PATCH(request: NextRequest) {
  try {
    const session = await auth.api.getSession({
      headers: await headers(),
    });

    if (!session?.user?.id) {
      return NextResponse.json(
        { error: "Unauthorized" },
        { status: 401 }
      );
    }

    const body = await request.json();
    const {
      aiProvider,
      aiModel,
      aiApiKey,
      aiBaseUrl,
      whisperModel,
    } = body;

    // Validate inputs
    const validProviders = ["local", "openai", "google", "anthropic"];
    if (aiProvider && !validProviders.includes(aiProvider)) {
      return NextResponse.json(
        { error: "Invalid aiProvider" },
        { status: 400 }
      );
    }

    if (aiBaseUrl && typeof aiBaseUrl !== "string") {
      return NextResponse.json(
        { error: "Invalid aiBaseUrl" },
        { status: 400 }
      );
    }

    const validWhisperModels = ["tiny", "base", "small", "medium", "large"];
    if (whisperModel && !validWhisperModels.includes(whisperModel)) {
      return NextResponse.json(
        { error: "Invalid whisperModel" },
        { status: 400 }
      );
    }

    const updatedUser = await prisma.user.update({
      where: { id: session.user.id },
      data: {
        ...(aiProvider !== undefined && { ai_provider: aiProvider }),
        ...(aiModel !== undefined && { ai_model: aiModel }),
        ...(aiApiKey !== undefined && { ai_api_key: aiApiKey }),
        ...(aiBaseUrl !== undefined && { ai_base_url: aiBaseUrl }),
        ...(whisperModel !== undefined && { whisper_model: whisperModel }),
      },
      select: {
        ai_provider: true,
        ai_model: true,
        ai_api_key: true,
        ai_base_url: true,
        whisper_model: true,
      },
    });

    return NextResponse.json({
      aiProvider: updatedUser.ai_provider,
      aiModel: updatedUser.ai_model,
      aiApiKey: updatedUser.ai_api_key,
      aiBaseUrl: updatedUser.ai_base_url,
      whisperModel: updatedUser.whisper_model,
    });
  } catch (error) {
    console.error("Error updating AI settings:", error);
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    );
  }
}