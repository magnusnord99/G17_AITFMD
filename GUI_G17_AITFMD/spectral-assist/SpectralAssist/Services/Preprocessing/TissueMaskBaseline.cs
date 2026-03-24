namespace SpectralAssist.Services.Preprocessing;

/// <summary>
/// Placeholder for tissue masking parity with Python <c>tissue_mask.build_tissue_mask</c>
/// (mean + Otsu + morphology + side selection). Implement here and add golden tests
/// that compare to <c>skimage</c> output.
/// </summary>
public static class TissueMaskBaseline
{
    // Future: bool[,] or byte[] mask H×W matching Python uint8 {0,1} saved masks.
}
