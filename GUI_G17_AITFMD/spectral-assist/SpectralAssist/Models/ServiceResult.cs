namespace SpectralAssist.Models;

public readonly struct ServiceResult<T>
{
    public T? Value { get; init; }
    public string? Error { get; init; }
    public bool IsSuccess => Error == null;

    public static ServiceResult<T> Ok(T value) => new() { Value = value };
    public static ServiceResult<T> Fail(string error) => new() { Error = error };
}
