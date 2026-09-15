#include "EmvBridgeSubsystem.h"

#include "Sockets.h"
#include "SocketSubsystem.h"
#include "IPAddress.h"
#include "Common/TcpSocketBuilder.h"
#include "HAL/RunnableThread.h"
#include "HAL/PlatformTime.h"
#include "Misc/CommandLine.h"
#include "Misc/Parse.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"
#include "Dom/JsonObject.h"

// ===========================================================================
//  Subsystem lifecycle
// ===========================================================================
void UEmvBridgeSubsystem::Initialize(FSubsystemCollectionBase& Collection)
{
	Super::Initialize(Collection);

	// Optional overrides:  -emvhost=127.0.0.1  -emvport=7777
	FString HostArg;
	if (FParse::Value(FCommandLine::Get(), TEXT("emvhost="), HostArg) && !HostArg.IsEmpty())
	{
		Host = HostArg;
	}
	int32 PortArg = 0;
	if (FParse::Value(FCommandLine::Get(), TEXT("emvport="), PortArg) && PortArg > 0)
	{
		Port = PortArg;
	}

	Runner = new FEmvBridgeRunnable(this, Host, Port);
	Thread = FRunnableThread::Create(Runner, TEXT("EmvBridge"), 0, TPri_AboveNormal);
	UE_LOG(LogTemp, Log, TEXT("[emv] bridge subsystem started, target %s:%d"), *Host, Port);
}

void UEmvBridgeSubsystem::Deinitialize()
{
	if (Runner)
	{
		Runner->Stop();
	}
	if (Thread)
	{
		Thread->Kill(true);   // waits for Run() to return
		delete Thread;
		Thread = nullptr;
	}
	if (Runner)
	{
		delete Runner;
		Runner = nullptr;
	}
	Super::Deinitialize();
}

// ===========================================================================
//  Thread-safe hand-offs
// ===========================================================================
void UEmvBridgeSubsystem::SetEvState(const FEmvEvState& State)
{
	FScopeLock Lock(&EvLock);
	PendingEv = State;
	bEvDirty = true;
}

bool UEmvBridgeSubsystem::TakeEvState(FEmvEvState& Out)
{
	FScopeLock Lock(&EvLock);
	if (!bEvDirty)
	{
		return false;
	}
	Out = PendingEv;
	bEvDirty = false;
	return true;
}

void UEmvBridgeSubsystem::PushNpcFrame(FEmvNpcFrame&& Frame)
{
	FScopeLock Lock(&FrameLock);
	// A reset re-seeds the whole population: never blend across it, and never
	// let stale pre-reset frames stay in the playback window.
	if (Frame.bReset)
	{
		FrameBuffer.Reset();
	}
	// Out-of-order or rewound sim time (a restarted brain) invalidates the ring.
	else if (FrameBuffer.Num() > 0 && Frame.SimTime <= FrameBuffer.Last().SimTime)
	{
		FrameBuffer.Reset();
	}

	FrameBuffer.Add(MoveTemp(Frame));
	while (FrameBuffer.Num() > MaxBufferedFrames)
	{
		FrameBuffer.RemoveAt(0);
	}
}

bool UEmvBridgeSubsystem::GetInterpolationFrames(FEmvNpcFrame& OutPrev, FEmvNpcFrame& OutNext) const
{
	FScopeLock Lock(&FrameLock);
	if (FrameBuffer.Num() == 0)
	{
		return false;
	}
	OutNext = FrameBuffer.Last();
	OutPrev = FrameBuffer.Num() > 1 ? FrameBuffer[FrameBuffer.Num() - 2] : OutNext;
	return true;
}

bool UEmvBridgeSubsystem::GetNewestSimTime(double& OutSimTime) const
{
	FScopeLock Lock(&FrameLock);
	if (FrameBuffer.Num() == 0)
	{
		return false;
	}
	OutSimTime = FrameBuffer.Last().SimTime;
	return true;
}

bool UEmvBridgeSubsystem::GetSpeedCap(double& OutCapMs) const
{
	FScopeLock Lock(&FrameLock);
	if (FrameBuffer.Num() == 0 || FrameBuffer.Last().SpeedCapMs < 0.0)
	{
		return false;
	}
	OutCapMs = FrameBuffer.Last().SpeedCapMs;
	return true;
}

bool UEmvBridgeSubsystem::GetFramesForSimTime(double RenderSimTime, FEmvNpcFrame& OutA,
	FEmvNpcFrame& OutB, float& OutAlpha) const
{
	FScopeLock Lock(&FrameLock);
	const int32 Num = FrameBuffer.Num();
	if (Num == 0)
	{
		return false;
	}
	if (Num == 1)
	{
		OutA = OutB = FrameBuffer[0];
		OutAlpha = 1.0f;
		return true;
	}

	// Behind the buffer (playback clock started cold): hold the oldest pair.
	if (RenderSimTime <= FrameBuffer[0].SimTime)
	{
		OutA = FrameBuffer[0];
		OutB = FrameBuffer[1];
		OutAlpha = 0.0f;
		return true;
	}
	// Ahead of the buffer (starved by a late arrival): hold the newest pose
	// rather than extrapolating into a guess that will have to be corrected.
	if (RenderSimTime >= FrameBuffer.Last().SimTime)
	{
		OutA = FrameBuffer[Num - 2];
		OutB = FrameBuffer[Num - 1];
		OutAlpha = 1.0f;
		return true;
	}

	for (int32 i = Num - 2; i >= 0; --i)
	{
		if (FrameBuffer[i].SimTime <= RenderSimTime)
		{
			OutA = FrameBuffer[i];
			OutB = FrameBuffer[i + 1];
			const double Span = OutB.SimTime - OutA.SimTime;
			OutAlpha = (Span > 1e-6)
				? static_cast<float>(FMath::Clamp((RenderSimTime - OutA.SimTime) / Span, 0.0, 1.0))
				: 1.0f;
			return true;
		}
	}

	OutA = FrameBuffer[0];
	OutB = FrameBuffer[1];
	OutAlpha = 0.0f;
	return true;
}

// ===========================================================================
//  Worker thread
// ===========================================================================
FEmvBridgeRunnable::FEmvBridgeRunnable(UEmvBridgeSubsystem* InOwner, const FString& InHost, int32 InPort)
	: Owner(InOwner), Host(InHost), Port(InPort)
{
}

bool FEmvBridgeRunnable::Init()
{
	return true;
}

void FEmvBridgeRunnable::Stop()
{
	bStop = true;
}

void FEmvBridgeRunnable::Exit()
{
	CloseSocket();
}

uint32 FEmvBridgeRunnable::Run()
{
	while (!bStop)
	{
		if (!EnsureConnected())
		{
			FPlatformProcess::Sleep(0.5f);   // retry the Python server
			continue;
		}
		PumpSend();
		PumpReceive();
		FPlatformProcess::Sleep(0.002f);     // ~500 Hz service loop
	}
	return 0;
}

bool FEmvBridgeRunnable::EnsureConnected()
{
	if (Socket)
	{
		return true;
	}
	ISocketSubsystem* SS = ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM);
	if (!SS)
	{
		return false;
	}
	TSharedRef<FInternetAddr> Addr = SS->CreateInternetAddr();
	bool bValidIp = false;
	Addr->SetIp(*Host, bValidIp);
	Addr->SetPort(Port);
	if (!bValidIp)
	{
		return false;
	}
	FSocket* NewSocket = SS->CreateSocket(NAME_Stream, TEXT("EmvBridgeClient"), Addr->GetProtocolType());
	if (!NewSocket)
	{
		return false;
	}
	if (!NewSocket->Connect(*Addr))
	{
		NewSocket->Close();
		SS->DestroySocket(NewSocket);
		return false;
	}
	NewSocket->SetNonBlocking(true);
	// Disable Nagle. Every EV message is a small partial segment, which Nagle
	// holds until the previous one is ACKed - up to a delayed-ACK period (~40 ms)
	// of latency added to a 20 ms control loop, and jitter on top of it.
	NewSocket->SetNoDelay(true);
	Socket = NewSocket;
	RecvAccum.Reset();
	if (Owner)
	{
		Owner->SetConnected(true);
	}
	UE_LOG(LogTemp, Log, TEXT("[emv] connected to brain %s:%d"), *Host, Port);
	return true;
}

void FEmvBridgeRunnable::CloseSocket()
{
	if (Socket)
	{
		ISocketSubsystem* SS = ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM);
		Socket->Close();
		if (SS)
		{
			SS->DestroySocket(Socket);
		}
		Socket = nullptr;
	}
	if (Owner)
	{
		Owner->SetConnected(false);
	}
}

void FEmvBridgeRunnable::PumpSend()
{
	const double Now = FPlatformTime::Seconds();
	if (Now - LastSendSeconds < SendPeriod)
	{
		return;
	}
	FEmvEvState Ev;
	if (!Owner || !Owner->TakeEvState(Ev))
	{
		return;
	}
	LastSendSeconds = Now;

	const FString Line = FString::Printf(
		TEXT("{\"t\":%.3f,\"seq\":%d,\"ev\":{\"x\":%.4f,\"y\":%.4f,\"vx\":%.4f,\"vy\":%.4f,\"yaw\":%.5f}}\n"),
		Now, SendSeq++, Ev.X, Ev.Y, Ev.Vx, Ev.Vy, Ev.Yaw);

	FTCHARToUTF8 Utf8(*Line);
	int32 Sent = 0;
	if (Socket && !Socket->Send(reinterpret_cast<const uint8*>(Utf8.Get()), Utf8.Length(), Sent))
	{
		CloseSocket();   // broken pipe -> reconnect on next loop
	}
}

void FEmvBridgeRunnable::PumpReceive()
{
	if (!Socket)
	{
		return;
	}
	uint8 Buffer[65536];
	int32 Read = 0;
	while (Socket && Socket->Recv(Buffer, sizeof(Buffer), Read, ESocketReceiveFlags::None) && Read > 0)
	{
		FUTF8ToTCHAR Conv(reinterpret_cast<const ANSICHAR*>(Buffer), Read);
		RecvAccum.AppendChars(Conv.Get(), Conv.Length());
	}

	int32 NewlineIdx = INDEX_NONE;
	while (RecvAccum.FindChar('\n', NewlineIdx))
	{
		const FString Line = RecvAccum.Left(NewlineIdx);
		RecvAccum = RecvAccum.RightChop(NewlineIdx + 1);
		if (!Line.IsEmpty())
		{
			ParseAndPush(Line);
		}
	}
}

bool FEmvBridgeRunnable::ParseAndPush(const FString& Line)
{
	TSharedPtr<FJsonObject> Root;
	const TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(Line);
	if (!FJsonSerializer::Deserialize(Reader, Root) || !Root.IsValid())
	{
		return false;
	}
	const TArray<TSharedPtr<FJsonValue>>* NpcArray = nullptr;
	if (!Root->TryGetArrayField(TEXT("npc"), NpcArray))
	{
		return false;
	}

	FEmvNpcFrame Frame;
	Frame.SimTime = Root->GetNumberField(TEXT("t"));
	Frame.Seq = static_cast<int32>(Root->GetNumberField(TEXT("seq")));
	Root->TryGetBoolField(TEXT("reset"), Frame.bReset);
	// Optional: only the SUMO modes send a speed ceiling.
	double Cap = -1.0;
	if (Root->TryGetNumberField(TEXT("vcap"), Cap))
	{
		Frame.SpeedCapMs = Cap;
	}
	Frame.WallRecvSeconds = FPlatformTime::Seconds();
	Frame.Npcs.Reserve(NpcArray->Num());

	for (const TSharedPtr<FJsonValue>& Value : *NpcArray)
	{
		const TSharedPtr<FJsonObject> Obj = Value->AsObject();
		if (!Obj.IsValid())
		{
			continue;
		}
		FEmvNpcState N;
		N.Id = static_cast<int32>(Obj->GetNumberField(TEXT("id")));
		N.X = Obj->GetNumberField(TEXT("x"));
		N.Y = Obj->GetNumberField(TEXT("y"));
		N.Yaw = Obj->GetNumberField(TEXT("yaw"));
		N.Vx = Obj->GetNumberField(TEXT("vx"));
		N.State = static_cast<int32>(Obj->GetNumberField(TEXT("s")));
		Frame.Npcs.Add(N);
	}

	if (Owner)
	{
		Owner->PushNpcFrame(MoveTemp(Frame));
	}
	return true;
}
