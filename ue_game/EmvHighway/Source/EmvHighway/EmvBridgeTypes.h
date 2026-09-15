#pragma once

#include "CoreMinimal.h"

// ---------------------------------------------------------------------------
// Data + coordinate contract for the emv <-> UE bridge.
// This is the C++ mirror of emv/ue/protocol.py. Keep the two in sync:
//   emv road frame (metres):  x forward, y lateral (+y = driver's LEFT),
//     lane 0 (rightmost) centre y=1.75, middle y=5.25, left y=8.75.
//   UE world (centimetres, left-handed, +Y to the right of +X): the highway is
//     laid straight along +X with the MIDDLE LANE centred on Y = 0.
// ---------------------------------------------------------------------------

namespace EmvCoords
{
	static constexpr double YCenterM = 5.25;   // middle-lane centre in the emv frame
	static constexpr double MToCm = 100.0;

	// emv (m) -> UE (cm) planar position
	FORCEINLINE FVector2D EmvToUeXY(double ExM, double EyM)
	{
		return FVector2D(ExM * MToCm, -(EyM - YCenterM) * MToCm);
	}

	// UE (cm) -> emv (m) planar position
	FORCEINLINE void UeToEmvXY(double UxCm, double UyCm, double& ExM, double& EyM)
	{
		ExM = UxCm / MToCm;
		EyM = -UyCm / MToCm + YCenterM;
	}

	// emv heading (rad, atan2(vy,vx)) -> UE yaw (deg about +Z)
	FORCEINLINE double EmvYawToUeDeg(double YawRad)
	{
		return -FMath::RadiansToDegrees(YawRad);
	}

	// UE yaw (deg) -> emv heading (rad)
	FORCEINLINE double UeYawToEmvRad(double YawDeg)
	{
		return -FMath::DegreesToRadians(YawDeg);
	}

	// velocity transforms like position: cm/s <-> m/s, Y sign-flipped
	FORCEINLINE void UeToEmvVel(double UvxCms, double UvyCms, double& VxMs, double& VyMs)
	{
		VxMs = UvxCms / MToCm;
		VyMs = -UvyCms / MToCm;
	}
}

// EV kinematics in the emv frame (sent UE -> Python).
struct FEmvEvState
{
	double X = 30.0;
	double Y = 7.0;
	double Vx = 0.0;
	double Vy = 0.0;
	double Yaw = 0.0;
};

// One NPC as received from Python (emv frame).
struct FEmvNpcState
{
	int32 Id = 0;
	double X = 0.0;
	double Y = 0.0;
	double Yaw = 0.0;
	double Vx = 0.0;
	int32 State = 0;   // 0 unaware, 1 noticed, 2 yielding, 3 hold
};

// A full NPC frame plus the wall-clock time it arrived (for interpolation).
struct FEmvNpcFrame
{
	double SimTime = 0.0;
	int32 Seq = -1;
	bool bReset = false;
	double WallRecvSeconds = 0.0;
	/** Speed ceiling for the EV in m/s, or < 0 when the brain sends none.
	 *  The SUMO modes send their car-following safe speed here: the player owns
	 *  the EV's position, so SUMO cannot brake it - the game brakes itself. */
	double SpeedCapMs = -1.0;
	TArray<FEmvNpcState> Npcs;
};
