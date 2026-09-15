#pragma once

#include "CoreMinimal.h"
#include "Subsystems/GameInstanceSubsystem.h"
#include "HAL/Runnable.h"
#include "HAL/ThreadSafeBool.h"
#include "EmvBridgeTypes.h"
#include "EmvBridgeSubsystem.generated.h"

class FSocket;
class UEmvBridgeSubsystem;

/**
 * Background socket worker. Owns the client connection so neither recv nor send
 * ever blocks the game thread. Connects (with retry) to the Python brain,
 * pushes parsed NPC frames into the subsystem, and drains the latest EV state
 * the game thread posted.
 */
class FEmvBridgeRunnable : public FRunnable
{
public:
	FEmvBridgeRunnable(UEmvBridgeSubsystem* InOwner, const FString& InHost, int32 InPort);

	virtual bool Init() override;
	virtual uint32 Run() override;
	virtual void Stop() override;
	virtual void Exit() override;

private:
	bool EnsureConnected();
	void CloseSocket();
	void PumpReceive();      // read bytes, split lines, parse frames
	void PumpSend();         // send the latest EV state (throttled)
	bool ParseAndPush(const FString& Line);

	UEmvBridgeSubsystem* Owner = nullptr;
	FSocket* Socket = nullptr;
	FString Host;
	int32 Port = 7777;

	FThreadSafeBool bStop = false;
	FString RecvAccum;
	int32 SendSeq = 0;
	double LastSendSeconds = 0.0;
	double SendPeriod = 1.0 / 60.0;   // cap EV upstream to ~60 Hz
};

/**
 * Single global bridge between the game and the Python force-model brain.
 * The EV pawn posts its state each tick (SetEvState); the NPC manager reads the
 * two most recent frames (GetInterpolationFrames) and interpolates between them.
 */
UCLASS()
class EMVHIGHWAY_API UEmvBridgeSubsystem : public UGameInstanceSubsystem
{
	GENERATED_BODY()

public:
	virtual void Initialize(FSubsystemCollectionBase& Collection) override;
	virtual void Deinitialize() override;

	/** Game thread -> worker: newest EV kinematics (emv frame). */
	void SetEvState(const FEmvEvState& State);

	/** Worker -> game thread: newest EV state to send (returns false if none new). */
	bool TakeEvState(FEmvEvState& Out);

	/** Worker -> subsystem: a freshly parsed NPC frame enters the buffer. */
	void PushNpcFrame(FEmvNpcFrame&& Frame);

	/** Game thread: the two most recent frames to interpolate between. */
	bool GetInterpolationFrames(FEmvNpcFrame& OutPrev, FEmvNpcFrame& OutNext) const;

	/** Game thread: brain SimTime of the newest buffered frame. */
	bool GetNewestSimTime(double& OutSimTime) const;

	/** Game thread: newest EV speed ceiling in m/s. False when the brain sends
	 *  none (the force-model mode keeps its own IDM layer, so it sends none). */
	bool GetSpeedCap(double& OutCapMs) const;

	/**
	 * Game thread: the two buffered frames bracketing RenderSimTime, plus the
	 * blend factor between them. Interpolating on the brain's own clock (which
	 * advances by exactly dt per frame) instead of on arrival wall-clock is what
	 * keeps playback smooth when frames arrive unevenly.
	 * Returns false only when the buffer is empty.
	 */
	bool GetFramesForSimTime(double RenderSimTime, FEmvNpcFrame& OutA,
		FEmvNpcFrame& OutB, float& OutAlpha) const;

	bool IsConnected() const { return bConnected; }
	void SetConnected(bool bIn) { bConnected = bIn; }

private:
	FString Host = TEXT("127.0.0.1");
	int32 Port = 7777;

	FRunnableThread* Thread = nullptr;
	FEmvBridgeRunnable* Runner = nullptr;

	mutable FCriticalSection EvLock;
	FEmvEvState PendingEv;
	bool bEvDirty = false;

	mutable FCriticalSection FrameLock;
	/** Newest-last ring of recent frames; the NPC manager plays back from a
	 *  point a little way inside it, so a late arrival never starves rendering. */
	TArray<FEmvNpcFrame> FrameBuffer;
	static constexpr int32 MaxBufferedFrames = 24;   // ~1.4 s at 16.7 Hz

	FThreadSafeBool bConnected = false;
};
