def rotate_array(nums: list, k: int) -> list:
    if not nums:
        return nums
    k = k % len(nums)
    return nums[-k:] + nums[:-k] if k else nums[:]
