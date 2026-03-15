def remove_duplicates(nums: list) -> list:
    if not nums:
        return []
    result = [nums[0]]
    for i in range(1, len(nums)):
        if nums[i] != nums[i - 1]:
            result.append(nums[i])
    return result
